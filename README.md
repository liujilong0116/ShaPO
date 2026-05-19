# ShaPO: Sharpness-aware Preference Optimization

**ShaPO** introduces sharpness-aware preference optimization for token-level and reward-level training. It first finds alignment-sensitive neurons via a linear **probe**, then applies a SAM-style update only on that subspace—boosting robustness to noisy labels and domain shift.

- 🔬 **Two variants**: Token-level ShaPO and Reward-level ShaPO
- 🧭 **Alignment-sensitive subspace**: parameter selection via probing
- 🪄 **Plug-and-play**: drop into any DPO-style post-training stack
- 🧱 **Backbones**: validated on Pythia-2.8B, LLaMA-3.2-3B, LLaMA-3-8B and Qwen2.5-7B

**Paper**: _Revisiting Robustness for LLM Safety Alignment
via Selective Geometry Control_ (accepted for presentation at **ICML 2026**)

---

## 📦 Installation

### 1) Create a clean Conda environment

```bash
conda create -n shapo python=3.10 -y
conda activate shapo
```

### 2) Install PyTorch (choose your CUDA / CPU build)

> Replace the command below to match your system: https://pytorch.org/get-started/locally/

```bash
# Example for CUDA 12.x (adjust if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3) Install project dependencies

```bash
pip install -r requirements.txt
```

---

## 📁 Project Layout

```
.
├── README.md
├── requirements.txt
├── train.py
├── train_probe_pku_30k.py
├── gen_probe_vector.py
├── preference_datasets.py
├── configs/
├── datasets/                 # place/prep datasets
├── judeg_eval/               # code for eval
└── classifier_output/        # probe outputs
```

---

## 🚀 Quick Start

> Below are the **exact run commands** you provided, organized by task.  
> Tip: Use `CUDA_VISIBLE_DEVICES=...` or FSDP configs as needed on multi-GPU nodes.

### 1) Train the classification head (probe)

Trains a linear probe to identify alignment-sensitive neurons (e.g., with **Jigsaw Toxic** labels).

```bash
python train_probe_pku_30k.py   --pretrained_model /xxx/models/meta-llama/Llama-3.2-3B   --epoch 10   --batch_size 256   --learning_rate 1e-5   --output_fp ./classifier_output/llama323_layer_-1_pku_30k_safety   --dataset pku   --save_model
```

### 2) Extract the probe vector

Generates the probe vector used to locate the sensitive subspace.

```bash
python gen_probe_vector.py
```

### 3) Qwen3-8B-Base SFT (supervised fine-tuning)

Baseline SFT run (Hydra-style arguments).

```bash
python train.py   model=llama323b   datasets=[pku_30k_harmless]   loss=sft   exp_name=LLaMA3-323B-SFT   gradient_accumulation_steps=2   batch_size=64   eval_batch_size=32   trainer=FSDPTrainer   sample_during_eval=false   model.fsdp_policy_mp=bfloat16
```

### 4) LLaMA3.2-3B **ShaPO (Token-level)**

Token-level ShaPO combines DPO with SAM-style perturbations on the identified subspace.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u train.py model=llama323b datasets=[pku_30k_harmless] loss=dpo loss.beta=0.1 exp_name=llama323_DrShaPO_pku_30k_harmless gradient_accumulation_steps=2 batch_size=32 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/home/y/yangyh/ljl/ShaPO/.cache/yangyh/llama38_pku_30k_harmless_sft_2026-01-01_23-20-03_192292/LATEST/policy.pt loss.name2=shapo warmup_steps=10 max_grad_norm=10 n_eval_examples=256 eval_every=2048 reward_beta=10 interval_for_shapo=5 if_output=false if_save=true probe_percentage=1 loss.mode_loss=DrDPO
```

### 5) LLaMA3.2-8B **ShaPO (Reward-level)**

Reward-level ShaPO anchors robustness to a reward model signal while staying RL-free.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u train.py model=llama323b datasets=[pku_30k_harmless] loss=dpo loss.beta=0.1 exp_name=llama38_DrShaPO_pku_30k_harmless gradient_accumulation_steps=2 batch_size=32 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/home/y/yangyh/ljl/ShaPO/.cache/yangyh/llama38_pku_30k_harmless_sft_2026-01-01_23-20-03_192292/LATEST/policy.pt loss.name2=shapo warmup_steps=10 max_grad_norm=10 n_eval_examples=256 eval_every=2048 reward_beta=10 interval_for_shapo=5 if_output=false if_save=true probe_percentage=1 loss.mode_loss=DrDPO use_reward=true
```

> **Note:** For single-GPU training, set `gradient_accumulation_steps=1`.
> The SHAPO perturbation/restoration logic is designed for FSDP multi-GPU training, and single-GPU runs with gradient accumulation may trigger inconsistent gradient states.

---

## 📚 Datasets

- **PKU-30K** (helpful & harmless preference pairs)

> Prepare or symlink them into `./data/` or update your config paths accordingly.

---

## ⚙️ Common Tips

- 🧮 **Precision**: `bfloat16` works well with FSDP; adjust for your hardware.
- 🧵 **FSDP**: Ensure PyTorch build and NCCL are compatible; set `NCCL_P2P_DISABLE=1` if you hit P2P issues.
- 💾 **Checkpoints**: `model.archive` should point to your SFT checkpoint (`.pt`) before launching ShaPO runs.
- 🧪 **Eval cadence**: Tune `eval_every` and `n_eval_examples` for your compute budget.

---
