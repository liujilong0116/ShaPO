#!/usr/bin/env python
# -*- coding: utf-8 -*-
import hydra

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig

from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port


#修改位置
def rank_value_vecs(model, toxic_vector: torch.Tensor, top_k: int = 128, similarity_threshold: float = None):
    """
    支持基于 top_k 或 similarity_threshold 的神经元筛选方式。
    返回 [(similarity, layer_idx, neuron_idx), ...]
    """
    print(f"[rank_value_vecs] 开始 — top_k={top_k}, similarity_threshold={similarity_threshold} jkljkljkljlkjlkjlk")
    model_type = getattr(model.config, "model_type", "")
    print(f"[rank_value_vecs] 模型类型: {model_type}")



    if getattr(model.config, "model_type", "").startswith("gpt2"):
        layers = model.transformer.h
    elif getattr(model.config, "model_type", "") in ("gpt_neox", "pythia"):
        layers = model.gpt_neox.layers
    elif getattr(model.config, "model_type", "") in ("llama", "qwen3", "qwen2"):
        layers = model.model.layers
    elif getattr(model.config, "model_type", "") in ("qwen2_vl"):
        layers = model.language_model.layers
    else:
        raise RuntimeError(f"Unsupported model_type={model.config.model_type}")
    print(f"[rank_value_vecs] 层数: {len(layers)}")
    device = next(model.parameters()).device
    tox = toxic_vector.to(next(model.parameters()).device)
    tox = tox / (tox.norm() + 1e-9)
    tox_norm = tox.norm().item() + 1e-9
    print(f"[rank_value_vecs] 毒性向量已归一化 (norm={tox_norm:.4f})，device={device}")
    candidates = []
    for layer_idx, layer in enumerate(layers):
        if hasattr(layer.mlp, "c_proj"):
            W = layer.mlp.c_proj.weight
        elif hasattr(layer.mlp, "dense_4h_to_h"):
            W = layer.mlp.dense_4h_to_h.weight
        elif hasattr(layer.mlp, "down_proj"):   # qwen2vl
            W = layer.mlp.down_proj.weight
        else:
            raise RuntimeError("Cannot find MLP output weight")
        
        # print(W.shape)
        if getattr(model.config, "model_type", "").startswith("gpt2"):
            W_norm = W.norm(dim=1)
            tox_norm = tox.norm()
            sims = torch.mv(W, tox) / (W_norm * tox_norm + 1e-9)
        elif getattr(model.config, "model_type", "") in ("gpt_neox", "pythia", "qwen2_vl", "llama", "qwen3", "qwen2"):
            W_norm = W.norm(dim=0)
            tox_norm = tox.norm()
            sims = torch.mv(W.T, tox) / (W_norm * tox_norm + 1e-9)
        

        for idx, sim in enumerate(sims.tolist()):
            if similarity_threshold is not None:
                if sim >= similarity_threshold:
                    
                    candidates.append((sim, layer_idx, idx))
            else:
                candidates.append((sim, layer_idx, idx))
            
    print(f"[rank_value_vecs] 总计收集到 {len(candidates)} 个候选项，准备排序，取出 {top_k} 个…")
    # candidates.sort(key=lambda x: x[0], reverse=True)
    candidates.sort(key=lambda x: x[0], reverse=False)

    if similarity_threshold is not None:
        return candidates
    else:
        # print(f"[rank_value_vecs] 使用 top_k，返回前 {len(candidates)} 项 (实际请求 {top_k})")
        return candidates[:top_k]

def get_setiv_vec(path=None, top_k=128):
    
    # model_id = "/root/autodl-tmp/models/openai-community/gpt2-medium"
    # probe_path = "/mnt/scratch/y/yangyh/ShaPO/classifier_output/toxic_gpt_medium.pt"

    # model_id = "/root/autodl-tmp/models/Qwen/Qwen2-VL-2B"
    # probe_path = "/mnt/scratch/y/yangyh/ShaPO/classifier_output/toxic_qwen2vl2b_12k.pt"
    # probe_path = "/mnt/scratch/y/yangyh/ShaPO/classifier_output/toxic_qwen2vl2b_wo_3000.pt"
    # probe_path = "/mnt/scratch/y/yangyh/ShaPO/classifier_output/toxic_qwen2vl2b_6000.pt"
    
    
    model_id = "/mnt/scratch/y/yangyh/models/EleutherAI/pythia-2.8b"
    probe_path = "/mnt/scratch/y/yangyh/ShaPO/classifier_output/pythia_pku/toxic_pythia_pku.pt"
    probe_path = "/mnt/scratch/y/yangyh/ShaPO/classifier_output/toxic_pythia28.pt"

    # model_id = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B"
    # probe_path = "/mnt/scratch/y/yangyh/ShaPO/classifier_output/toxic_llama38.pt"

    # model_id = "/root/autodl-tmp/models/Qwen/Qwen3-4B"
    # probe_path = "/mnt/scratch/y/yangyh/ShaPO/classifier_output/qwen34b/toxic_qwen34b.pt"

    model_id = "/mnt/scratch/y/yangyh/models/Qwen/Qwen2.5-7B"
    probe_path = "/home/y/yangyh/ljl/ShaPO/classifier_output/qwen257_layer_-1_pku_30k_safety/toxic_pythia_pku_harmless.pt"

    model_id = "/mnt/scratch/y/yangyh/models/LLM-Research/Meta-Llama-3-8B"
    probe_path = "/home/y/yangyh/ljl/ShaPO/classifier_output/llama38_layer_-1_pku_30k_safety/toxic_pythia_pku_harmless.pt"

    model_id = "/mnt/scratch/y/yangyh/models/meta-llama/Llama-3.2-3B"
    probe_path = "/home/y/yangyh/ljl/ShaPO/classifier_output/llama323_layer_-1_pku_30k_safety/toxic_pythia_pku_harmless.pt"


    print(f"🔄 正在加载模型{model_id}")
    if "VL" in model_id and "Qwen" in model_id:
        from transformers import Qwen2VLModel
        model = Qwen2VLModel.from_pretrained(
                model_id,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id
        )
    print(model)
    print("✅ 模型加载完毕")
    # print(f"🔄 正在加载分词器{model_id}")
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # print("✅ 分词器加载完毕")
    model.eval()
    disable_dropout(model)
    
    if path:
        probe_path = path
    print("🛠️ 已禁用 Dropout")
    print(f"🔄 读取毒性向量：{probe_path}")
    # qwen2-vl-2b 使用 3000 * 2 个多模态 DPO 数据训练得到的毒性探针
    
    toxic_vector = torch.load(probe_path, map_location="cpu")
    

    top_vecs = rank_value_vecs(model, toxic_vector, top_k=top_k)
    return top_vecs

def get_setiv_vec_by_percentage(path=None, percentage=1):
    model_id = "/mnt/scratch/y/yangyh/models/meta-llama/Llama-3.2-3B"
    probe_path = "/home/y/yangyh/ljl/SharPO/classifier_output/llama323_layer_-1_pku_30k_safety/toxic_llama323_pku_harmless.pt"


    print(f"🔄 正在加载模型{model_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id
    )

    print("✅ 模型加载完毕")

    model.eval()
    disable_dropout(model)
    
    if path:
        probe_path = path
    print("🛠️ 已禁用 Dropout")
    print(f"🔄 读取毒性向量：{probe_path}")
    
    toxic_vector = torch.load(probe_path, map_location="cpu")
    total_neurons = 327680
    top_k = int(total_neurons * percentage / 100)
    top_k = max(1, min(top_k, total_neurons))

    top_vecs = rank_value_vecs(model, toxic_vector, top_k=top_k)
    return top_vecs

@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def main(cfg:DictConfig):
    print(f"🔄 正在加载模型 {cfg.model.name_or_path} …")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path, low_cpu_mem_usage=True
    )
    print("✅ 模型加载完毕")
    print(f"🔄 正在加载分词器 {cfg.model.name_or_path} …")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
    print("✅ 分词器加载完毕")
    model.eval()
    disable_dropout(model)
    print("🛠️ 已禁用 Dropout")
    print(f"🔄 读取毒性向量：{cfg.probe_path} …")
    toxic_vector = torch.load(cfg.probe_path, map_location="cpu")
    top_vecs = rank_value_vecs(model, toxic_vector, top_k=128)
    print(top_vecs)

if __name__ == "__main__":
    main()
