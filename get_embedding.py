import transformers
import torch
from preference_datasets import get_batch_iterator
from utils import disable_dropout
from typing import List, Dict, Optional, Tuple
import numpy as np
import tqdm

def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

def ensure_tokenized(
    batch: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int = 512,
    only_response: bool = False
) -> Dict[str, torch.Tensor]:

    chosen_enc = tokenizer(
        batch['chosen'] if not only_response else batch["chosen_response_only"],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    rejected_enc = tokenizer(
        batch['rejected'] if not only_response else batch["rejected_response_only"],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return {
                "input_ids": chosen_enc["input_ids"], 
                "attention_mask": chosen_enc["attention_mask"]
            }, {
                "input_ids": rejected_enc["input_ids"], 
                "attention_mask": rejected_enc["attention_mask"],
            }

def pool_hidden_states(
    last_hidden: torch.Tensor,           # [B, T, H]
    attention_mask: torch.Tensor,        # [B, T]
    tokenizer: transformers.PreTrainedTokenizer,
    pooling: str = "last_non_pad",
    input_ids: Optional[torch.Tensor] = None  # [B, T], 仅 'eos' 策略需要
) -> torch.Tensor:
    """
    将最后一层 hidden states 做序列级 pooling，得到 [B, H]。
    """
    assert last_hidden.dim() == 3 and attention_mask.dim() == 2
    B, T, H = last_hidden.shape

    if pooling == "mean":
        # 按 mask 求均值
        lengths = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B, 1]
        summed = (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1)  # [B, H]
        emb = summed / lengths
        return emb

    if pooling == "eos":
        # 取每个样本中最后一个 eos_token 的位置；若不存在，则回退到 last_non_pad
        assert input_ids is not None, "eos pooling 需要 input_ids"
        eos_id = tokenizer.eos_token_id
        # 将非 eos 的位置置为 -inf，找到每行最后一次出现 eos 的索引
        is_eos = (input_ids == eos_id) & (attention_mask == 1)
        # 若某行没有 eos，全 False，下面回退
        idxs = []
        for b in range(B):
            pos = torch.nonzero(is_eos[b], as_tuple=False).flatten()
            if pos.numel() == 0:
                # 回退到最后一个非 pad
                last_idx = int(attention_mask[b].sum().item()) - 1
                idxs.append(last_idx)
            else:
                idxs.append(int(pos[-1].item()))
        idxs = torch.tensor(idxs, device=last_hidden.device, dtype=torch.long)  # [B]
        emb = last_hidden[torch.arange(B, device=last_hidden.device), idxs]     # [B, H]
        return emb

    # 默认：取最后一个非 pad token 的向量
    last_indices = attention_mask.sum(dim=1) - 1  # [B]
    last_indices = last_indices.clamp(min=0).to(torch.long)
    emb = last_hidden[torch.arange(B, device=last_hidden.device), last_indices]  # [B, H]
    return emb


model_path = "/root/autodl-tmp/models/EleutherAI/pythia-2.8b"

only_response = False
output_dir = "/root/autodl-tmp/SharPO/pythia_hh_embedding_output"

only_response = True
output_dir = "/root/autodl-tmp/SharPO/pythia_hh_embedding_output_only_response"

method_name = "SFT"
stata_path = "/root/autodl-tmp/SharPO/.cache/root/anthropic_sft_pythia28/pythia28_2025-09-12_04-47-10_799414/LATEST/policy.pt"

method_name = "DPO"
stata_path = "/root/autodl-tmp/SharPO/.cache/root/DPO_pythia28_hh_2025-09-21_03-27-41_883891/LATEST/policy.pt"

method_name = "ShaPO_same"
stata_path = "/root/autodl-tmp/SharPO/.cache/root/SharPO_pythia28_hh_same_2025-09-21_03-30-17_380974/LATEST/policy.pt"

# method_name = "ShaPO_more"
# stata_path = "/root/autodl-tmp/SharPO/.cache/root/SharPO_pythia28_hh_harmless_more_2025-09-21_02-07-54_511461/LATEST/policy.pt"qu


policy = transformers.AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, 
    device_map="auto"
    )
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
state_dict = torch.load(stata_path, map_location='cpu')
policy.load_state_dict(state_dict['state'],strict=False)
disable_dropout(policy)


data_iterator_kwargs = dict(
    names=["hh"],
    tokenizer=tokenizer,
    shuffle=True,
    max_length=512,
    max_prompt_length=256,
    sft_mode=False,
)
train_batch = list(get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=1, n_examples=None, batch_size=32, silent=True, harmless_rate=0.7))
eval_batch = list(get_batch_iterator(**data_iterator_kwargs, split='test', n_epochs=None, n_examples=256, batch_size=32, silent=True, harmless_rate=0.7))

shp_data_iterator_kwargs = dict(
    names=["shp"],
    tokenizer=tokenizer,
    shuffle=True,
    max_length=512,
    max_prompt_length=256,
    sft_mode=False,
)
shp_batch = list(get_batch_iterator(**shp_data_iterator_kwargs, split='test', n_epochs=None, n_examples=256, batch_size=32, silent=True, harmless_rate=0.7))

def get_all_embeddings(batch):
    all_chosen_embeds, all_rejected_embeds = [], []
    for item in tqdm.tqdm(batch):
        # print(item["chosen"][0], item["rejected"][0])
        chosen_pack, rejected_pack = ensure_tokenized(item, tokenizer, max_length=512, only_response=only_response)
        chosen_pack = batch_to_device(chosen_pack, device=policy.device)
        rejected_pack = batch_to_device(rejected_pack, device=policy.device)
        def get_embedding(pack):
            output = policy(
                    **pack,
                    output_hidden_states=True,
                    use_cache=False  # 关掉 KV cache 更省显存
                )
            last_hidden = output.hidden_states[-1]
            embs = pool_hidden_states(
                    last_hidden=last_hidden,
                    attention_mask=pack["attention_mask"],
                    tokenizer=tokenizer,
                    pooling=None,
                    input_ids=pack["input_ids"]
                )  # [B, H]
            return embs.float().cpu().detach().numpy()
        all_chosen_embeds.append(get_embedding(chosen_pack))
        all_rejected_embeds.append(get_embedding(rejected_pack))
    return all_chosen_embeds, all_rejected_embeds


all_chosen_embeds, all_rejected_embeds = get_all_embeddings(train_batch[:8])
chosen_arr = np.concatenate(all_chosen_embeds, axis=0)
rejected_arr = np.concatenate(all_rejected_embeds, axis=0)
print(chosen_arr.shape, rejected_arr.shape)
np.save(f"{output_dir}/{method_name}_train_chosen_embeds.npy", chosen_arr)
np.save(f"{output_dir}/{method_name}_train_rejecte_embeds.npy", rejected_arr)

all_chosen_embeds, all_rejected_embeds = get_all_embeddings(eval_batch)
chosen_arr = np.concatenate(all_chosen_embeds, axis=0)
rejected_arr = np.concatenate(all_rejected_embeds, axis=0)
print(chosen_arr.shape, rejected_arr.shape)
np.save(f"{output_dir}/{method_name}_eval_chosen_embeds.npy", chosen_arr)
np.save(f"{output_dir}/{method_name}_eval_rejecte_embeds.npy", rejected_arr)

all_chosen_embeds, all_rejected_embeds = get_all_embeddings(shp_batch)
chosen_arr = np.concatenate(all_chosen_embeds, axis=0)
rejected_arr = np.concatenate(all_rejected_embeds, axis=0)
print(chosen_arr.shape, rejected_arr.shape)
np.save(f"{output_dir}/{method_name}_shp_chosen_embeds.npy", chosen_arr)
np.save(f"{output_dir}/{method_name}_shp_rejecte_embeds.npy", rejected_arr)


# CUDA_VISIBLE_DEVICES=0 python get_embedding.py