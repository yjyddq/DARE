<p align="center">
  <img src="assets/DARE_logo.png" style="width:60%; height:auto;">
</p>

<div align="center">

<h2>DARE: dLLM Alignment and Reinforcement Executor</h2>

[![Paper](https://img.shields.io/badge/Paper-arXiv:2604.04215-red)](https://arxiv.org/pdf/2604.04215)
<img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
<img src="https://visitor-badge.laobi.icu/badge?page_id=yjyddq.DARE" />
<img src="https://img.shields.io/github/stars/yjyddq/DARE?style=flat-square&logo=github&label=Stars" alt="Stars">
<img src="https://img.shields.io/github/issues/yjyddq/DARE?color=red&label=Issues" alt="Open Issues">
<img src="https://img.shields.io/github/issues-closed/yjyddq/DARE?color=success&label=Issues" alt="Closed Issues">

</div>


## 🎯 Overview

We introduce **DARE** (**d**LLM **A**lignment and **R**einforcement **E**xecutor), a flexible and efficient supervised-finetuning (SFT) and reinforcement learning (RL) training framework designed specifically for diffusion large language models (dLLMs). DARE also integrates dLLMs into a comprehensive evaluation platform. It aims to be both flexible and user-friendly to use with:
- Easy extension of diverse RL algorithms for dLLMs
- Easy extension of extra benchmark evaluations for dLLMs
- Easy integration of popular and upcoming dLLM infras and HuggingFace weights

DARE is a work in progress, we plan to support more models and algorithm for training and evaluation. **We warmly welcome the research community to collaborations, give feedback and share suggestions.** Let's advance the diffusion large language models together !!!👊

<!-- **Optimization Plan in RL Pipeline**
> [!TIP]
> For MDLMs (LLaDA or Dream), we decouple the attention backend used during training from that used during rollout. Rollout uses `flash_attn_func` or `flash_attn_with_kvcache` for KV-cache, training adopts `flash_attn_varlen_func` to skip meaningless computation on padding tokens. The entire pipeline speed-up approximately ⚡️ **4×**.

<p align="center">
  <img src="assets/optimization_plan_mdlm.png" style="width:85%; height:auto;">
</p>

> [!TIP]
> For BDLMs like SDAR: We rollout with compatible lmdeploy inference and adopt SDAR's logits-free `fused_linear_cross_entropy` to cut memory usage, enable online weights update for rollout policy. The entire pipeline will be accelerated more than ⚡️ **14×**.

<p align="center">
  <img src="assets/optimization_plan_bdlm.png" style="width:85%; height:auto;">
</p> -->

## 📢 News
- [2026-07-16]: Support ESPO for LLaDA.
- [2026-07-15]: Support JustGRPO for LLaDA.
- [2026-06-12]: Fix DLLM sampling params for rollout diversity [SGLang branch](https://github.com/sgl-project/sglang/pull/27943)
- [2026-04-19]: Support ebpo for SDAR.
- [2026-04-18]: Add an example of multi nodes rl training for LLaDA d1.
- [2026-04-05]: We release our report "DARE: Diffusion Large Language Models Alignment and Reinforcement Executor" on arXiv.
- [2026-03-23]: Support d-treerpo algorithm for LLaDA and Dream.
- [2026-03-16]: Support bgpo and ebpo for LLaDA2.X.
- [2026-03-13]: Fix bugs in SDAR SGLang rollout and dp actor.
- [2026-03-12]: Support sp for SDAR family, LLaDA2.0 and 2.1.
- [2026-03-10]: Support evaluation of SDAR-30B-A3B-Chat with SGLang and lmdeploy.
- [2026-03-09]: Support evaluation of LLaDA2.1-mini with SGLang.
- [2026-03-03]: Support mdpo for LLaDA and Dream.
- [2026-03-02]: Support SGLang for SDAR rl rollout.
- [2026-02-28]: Fix bugs and update features for LLaDA and Dream sequence parallel.
- [2026-02-27]: Support evaluation of SDAR with SGLang.
- [2026-02-26]: Update LLaDA cj-grpo and add Dream cj-grpo.

<details>
<summary>2025</summary>

- [2025-12-28]: Fix bugs and update features in dp_actor_algorithm.
- [2025-12-24]: Support online rl (online weight update of rollout) for SDAR.
- [2025-12-23]: Support vrpo (preference optimization) for Dream.
- [2025-12-16]: Support vrpo (preference optimization) for LLaDA.
- [2025-12-12]: Support sft and peft of SDAR.
- [2025-12-11]: Support evaluation of LLaDAMoE and LLaDA2.0-mini with SGLang.
- [2025-12-08]: Support coupled-grpo, cj-grpo and spg algorithm.
- [2025-12-03]: Support sequence parallel to enable longer generation ability for dLLMs.
- [2025-12-01]: We initialize the codebase of DARE (dLLM Alignment and Reinforcement Executor), including training (sft, peft, and rl) and faster evaluation.

</details>


## 🔍 Catalogue

- [🎯 Overview](#-overview)
- [🏆 Key Features](#-key-features)
- [🛠️ Installation and Setup](#️-installation-and-setup)
- [🏋️ Training](#️-training)
- [📊 Evaluation](#-evaluation)
- [📈 Performance](#-performance)
- [📦 Supported Models](#-supported-models)
- [🌱 Supported RL Algorithms](#-supported-rl-algorithms)
- [📧 Contact](#-contact)
- [📚 Citation](#-citation)
- [🙏 Acknowledgments](#-acknowledgments)


## 🏆 Key Features

- **Algorithm Zoo**
  - Support d1, Coupled-GRPO, VRPO, MDPO, CJ-GRPO, BGPO, SPG, EBPO, d-TreeRPO, JustGRPO, ESPO
- **Model Diversity**
  - Masked diffusion language models (e.g., LLaDA and Dream)
  - Block diffusion language model (e.g., SDAR and LLaDA2.X)
- **Comprehensive Evaluation for dLLMs**
  - Integrate faster dLLM evaluation in [opencompass](https://github.com/open-compass/opencompass)
- **Upcoming Features**
  - Support multi-modal, omni, more algorithms


## 🛠️ Installation and Setup

Our training framework is built on top of [verl](https://github.com/volcengine/verl), providing a robust foundation for supervised finetuning and reinforcement learning experiments, and our evaluation framework is built on the top of [opencompass](https://github.com/open-compass/opencompass), providing a comprehensive and fast evaluations.

> [!NOTE]
> Due to some **irreconcilable dependency conflicts** between packages, we **strongly recommend using two separate virtual environments**, for training and evaluation, respectively.


### 🚀 Quick Installation

Clone the DARE repo:
```bash
git clone https://github.com/yjyddq/DARE
```

Build training vitual environment:

```bash
# Create and activate environment
conda create -n DARE python=3.10 -y
conda activate DARE

# Install dependencies
cd DARE
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
# or (Recommend)
# install from whl
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# You can go to https://flashattn.dev/ to find the compiled FlashAttention .whl that corresponds to your Python, PyTorch, and CUDA versions.
# After download the .whl, install it
pip install flash_attn-2.8.x+cu12xtorch2.x-cp3xx-cp3xx-linux_x86_64.whl # For example
```

Build evaluation vitual environment:

```bash
# Create and activate environment
conda create --name opencompass python=3.10 -y
conda activate opencompass

# Install dependencies
cd DARE/opencompass
pip install -e .

# For HumanEval evaluation, install the additional dependency:
git clone https://github.com/open-compass/human-eval.git
cd human-eval && pip install -e .
cd ..

# For Math evaluation, pip install the additional dependency:
pip install math_verify latex2sympy2_extended

## Full installation (with support for more datasets)
# pip install "opencompass[full]"

## Environment with model acceleration frameworks
# pip install "opencompass[lmdeploy]"
# or
# pip install lmdeploy==0.10.1
```

Install SGLang (required for SDAR and LLaDA2.x rollout and evaluation acceleration) from source. We recommend using the [SGLang PR branch](https://github.com/sgl-project/sglang/pull/27943) before it is merged, since it fixes DLLM sampling parameters for better rollout diversity:

```bash
# Recommended: use the DARE-compatible SGLang PR branch
git clone https://github.com/sgl-project/sglang.git
cd sglang
git fetch origin pull/27943/head:dllm-sampling-params
git checkout dllm-sampling-params

# Fallback: use the previous compatible release branch only if you do not need this PR fix
# git clone -b v0.5.9 https://github.com/sgl-project/sglang.git
# cd sglang

# Install the python packages
pip install --upgrade pip
pip install -e "python"
```

### 🔧 Model Setup

After downloading [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct), replace the source files with our modified versions to enable several key features:

```bash
# Copy modified files to your LLaDA model directory
cp models/xxx/* <path_to_llada_model>/
```

Or you can move the model weights (`.safetensors`) to `model/xxx/*`

```bash
# Copy weights to models/xxx/ directory
cp <path_to_llada_model>/*.safetensors models/xxx/
```

Also for Dream, SDAR, etc.

> [!NOTE]
> Since optimization plan in RL pipeline (various attention-computation backend), this step is indispensable.


### 🗄️ Dataset Setup

Preprocessed datasets is under `data/preprocessed`. Please refer `verl.utils.preprocess` to organize datasets. 


## 🏋️ Training

### 🚀 SFT Quick Start

```bash
bash scripts/run_sft.sh # | scripts/run_sft_peft.sh
```

Alternatively, use/write scripts in recipe/xxx/run_xxx.sh

```bash
# peft for llada_8b_instruct
bash recipe/run_sft_peft_llada_8b_instruct.sh 

# sft for dream_7b_instruct
bash recipe/run_sft_dream_7b_instruct.sh 

# peft for sdar_8b_chat
bash recipe/run_sft_peft_sdar_8b_chat.sh 
```

### 🚀 RL Quick Start

```bash
# online rl for llada_8b_instruct
bash recipe/run_d1_llada_8b_instruct.sh --task math # use Fast-dLLM for rollout acceleration

# online rl for dream_7b_instruct
bash recipe/run_coupled_grpo_dream_7b_instruct.sh --task math # use Fast-dLLM for rollout acceleration

# online rl for sdar_8b_chat
bash recipe/run_bgpo_sdar_8b_chat.sh --task math # use lmdeploy engine for rollout acceleration

# d-TreeRPO for llada_8b_instruct
bash recipe/llada/run_dtreerpo_llada_8b_instruct.sh --task math

# d-TreeRPO for dream_7b_instruct
bash recipe/dream/run_dtreerpo_dream_7b_instruct.sh --task math
```

### 🚀 DPO/VRPO Quick Start

Run an example for preference optimization. First download [argilla/ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned), then run `scripts/preprocess_dpo_dataset.sh` to save `ultrafeedback.parquet` under `data/preprocessed/dpo/train` and `data/preprocessed/dpo/test`

```bash
# preference optimization for llada_8b_instruct
bash recipe/run_vrpo_llada_8b_instruct.sh --task ultrafeedback

# preference optimization for dream_7b_instruct
bash recipe/run_vrpo_dream_7b_instruct.sh --task ultrafeedback
```

### 🚀 Convert Checkpoints Weights to HuggingFace Safetensors

```bash
# convert FSDP sharded checkpoints to HuggingFace safetensors format
bash scripts/convert_ckpt_to_hf.sh \
    --model_path models/LLaDA-8B-Instruct \
    --ckpt_path ./ckpts/<project>/<exp_name>/global_step_xxx/actor \
    --output_path ./converted_models/llada_8b_stepxxx \
    --model_name llada \
    --fsdp_strategy fsdp2 \
    --n_gpus 8
```


## 📊 Evaluation

### 🚀 Eval on OpenCompass Quick Start

First, please follow [opencompass](https://github.com/open-compass/opencompass) for benchmark dataset preparation. Then, you need to specify the model path in `opencompass/opencompass/configs/models/dllm/*`. For example `llada_instruct_8b.py`:

```bash
from opencompass.models import LLaDAModel

models = [
    dict(
        type=LLaDAModel,
        abbr='llada-8b-instruct',
        path='/TO/YOUR/PATH', # Need to modify
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
```

Evaluation of LLaDA-8B-Instruct on mmlu with hf backend:

```bash
bash scripts/eval_llada.sh --task mmlu
```

Evaluation of SDAR-8B-Chat on mmlu with lmdeploy backend:

```bash
bash scripts/eval_sdar_8b_chat.sh --task mmlu --engine lmdeploy
```

### 🚀 Eval on Local Bench Quick Start

If you want to add more benchmarks, models, or custom datasets, please refer to the [Evaluation Guideline](https://github.com/yjyddq/DARE/blob/main/opencompass/README.md).



## 📦 Supported Models

| Model | Params | Training Support | Evaluation Support | Inference Acceleration |
|-------|------------|------------------|--------------------|-------------------------------|
| **LLaDA-8B-Base** | 8B | sft/rl | ✅ | hf [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) |
| **LLaDA-8B-Instruct** | 8B | sft/rl | ✅ | hf [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) |
| **LLaDA-1.5** | 8B | sft/rl | ✅ | hf [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) |
| **Dream-7B-Instruct** | 7B | sft/rl | ✅ | hf [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) |
| **SDAR-1.7B-Chat** | 1.7B | sft/rl | ✅ | [lmdeploy](https://github.com/InternLM/lmdeploy) [SGLang](https://github.com/sgl-project/sglang) |
| **SDAR-4B-Chat** | 4B | sft/rl | ✅ | [lmdeploy](https://github.com/InternLM/lmdeploy) [SGLang](https://github.com/sgl-project/sglang) |
| **SDAR-8B-Chat** | 8B | sft/rl | ✅ | [lmdeploy](https://github.com/InternLM/lmdeploy) [SGLang](https://github.com/sgl-project/sglang) |
| **SDAR-30B-A3B-Chat** | 30BA3B | sft | ✅ | [lmdeploy](https://github.com/InternLM/lmdeploy) [SGLang](https://github.com/sgl-project/sglang) |
| **LLaDA2.0-mini** | 16BA1B | sft/rl | ✅ | [SGLang](https://github.com/sgl-project/sglang) |
| **LLaDA2.1-mini** | 16BA1B | sft/rl | ✅ | [SGLang](https://github.com/sgl-project/sglang) |


## 🌱 Supported RL Algorithms

| Algorithm | Arxiv | Source Code |
|-------|------------|------------------------|
| **d1** | [2504.12216](https://arxiv.org/pdf/2504.12216) | [dllm-reasoning/d1](https://github.com/dllm-reasoning/d1) |
| **vrpo** | [2505.19223](https://arxiv.org/abs/2505.19223) | [ML-GSAI/LLaDA-1.5](https://github.com/ML-GSAI/LLaDA-1.5) (closed source) |
| **coupled-grpo** | [2506.20639](https://arxiv.org/pdf/2506.20639) | [apple/ml-diffucoder](https://github.com/apple/ml-diffucoder) |
| **mdpo** | [2508.13148](https://arxiv.org/pdf/2508.13148) | [autonomousvision/mdpo](https://github.com/autonomousvision/mdpo) |
| **cj-grpo** | [2509.23924](https://arxiv.org/pdf/2509.23924) | [yjyddq/EOSER-ASS-RL](https://github.com/yjyddq/EOSER-ASS-RL) |
| **spg** | [2510.09541](https://arxiv.org/pdf/2510.09541) | [facebookresearch/SPG](https://github.com/facebookresearch/SPG) |
| **bgpo** | [2510.11683](https://arxiv.org/pdf/2510.11683) | [THU-KEG/BGPO](https://github.com/THU-KEG/BGPO) |
| **ebpo** | [2602.08676](https://arxiv.org/pdf/2602.08676) | [inclusionAI/LLaDA2.X](https://github.com/inclusionAI/LLaDA2.X) (closed source) |
| **d-treerpo** | [2512.09675](https://arxiv.org/pdf/2512.09675) | [THU-BPM/d-TreeRPO](https://github.com/THU-BPM/d-TreeRPO) |


## 📈 Performance

**Evaluation Result Reproduction**

|   Bench/Model   | LLaDA-8B | Dream-7B | SDAR-8B-Chat | SDAR-30B-A3B | LLaDA2.0-mini | LLaDA2.1-mini |
|-----------------|----------|----------|--------------|--------------|---------------|---------------|
|     **MMLU**    | 65.24 | 66.83 | 77.23 | 79.16 | 72.54 | 69.91 |
|   **MMLU-Pro**  | 36.82 | 31.89 | 56.49 | 25.59 | 57.10 | 57.52 |
|  **Hellaswag**  | 75.30 | 63.23 | 87.59 | 92.81 | 82.35 | 78.00 |
|    **ARC-C**    | 87.80 | 81.36 | 86.78 | 78.98 | 85.76 | 83.39 |
|    **GSM8k**    | 79.68 | 83.24 | 91.36 | 92.49 | 88.48 | 86.13 |
|     **MATH**    | 41.08 | 48.02 | 78.40 | 68.56 | 81.50 | 84.56 |
|     **GPQA**    | 31.82 | 26.77 | 41.40 | 36.36 | 34.34 | 34.34 |
|    **AIME24**   | 2.08  | 0.83  | 13.33 | 13.33 | 16.67 | 26.67 |
|    **AIME25**   | 0.42  | 0.00  | 16.67 | 6.67  | 23.33 | 26.67 |
|**OlympiadBench**| 9.70  | 12.22 | 24.93 | 32.90 | 38.82 | 40.31 |
|  **HumanEval**  | 46.34 | 78.05 | 79.88 | 84.15 | 81.10 | 81.10 |
|     **MBPP**    | 38.80 | 56.40 | 71.60 | 52.00 | 64.80 | 62.60 |

**Algorithm Comparison (LLaDA-8B-Instruct)**

|   Bench/Algo  | Baseline |    d1    |  Coupled-GRPO   | VRPO | CJ-GRPO | SPG  | BGPO |
|---------------|----------|----------|-----------------|------|---------|------|------|
|               |          |          | **Mathematics** |      |         |      |      |
| **GSM8k**     |   76.5   |   83.7   |       85.3      | 81.9 |   85.6  | 83.5 | 82.3 |
| **MATH**      |   34.6   |   40.6   |       41.0      | 35.8 |   39.2  | 40.6 | 40.0 |
|               |          |          |    **Coding**   |      |         |      |      |
| **HumanEval** |   46.9   |   47.6   |       45.1      | 52.4 |   45.1  | 48.8 | 45.1 |
| **MBPP**      |   37.9   |   39.1   |       38.1      | 42.8 |   40.9  | 41.9 | 40.3 |
|               |          |          |   **Planning**  |      |         |      |      |
| **Countdown** |   16.8   |   10.7   |       77.9      | 21.5 |   65.2  | 10.1 | 10.0 |
| **Sudoku**    |   26.2   |   31.8   |       21.3      | 29.0 |   25.0  | 27.9 | 42.6 |

**Algorithm Comparison (Dream-7B-Instruct)**

|   Bench/Algo  | Baseline |    d1    |  Coupled-GRPO   | CJ-GRPO | SPG  | BGPO |
|---------------|----------|----------|-----------------|---------|------|------|
|               |          |          | **Mathematics** |         |      |      |
| **GSM8k**     |   77.2   |   82.5   |       80.3      |   85.7  | 59.4 | 83.9 |
| **MATH**      |   39.6   |   49.7   |       40.4      |   50.7  | 25.2 | 48.9 |
|               |          |          |    **Coding**   |         |      |      |
| **HumanEval** |   57.9   |   60.7   |       61.6      |   58.5  | 17.7 | 56.7 |
| **MBPP**      |   56.2   | 56.5     |       60.3      |   57.5  | 54.4 | 58.7 |


## 📧 Contact

For any questions or collaboration inquiries, feel free to reach out Jingyi Yang at: [yangjingyi946@gmail.com](yangjingyi946@gmail.com).


## 👷‍♂️ Contributor

Waiting for your joining and contribution.



## 📚 Citation

If you find our work useful, please consider citing:

```bibtex
@article{yang2026dare,
  title={DARE: Diffusion Large Language Models Alignment and Reinforcement Executor},
  author={Yang, Jingyi and Jiang, Yuxian and Hu, Xuhao and Cheng, Shuang and Qi, Biqing and Shao, Jing},
  journal={arXiv preprint arXiv:2604.04215},
  year={2026}
}

@article{yang2025taming,
  title={Taming Masked Diffusion Language Models via Consistency Trajectory Reinforcement Learning with Fewer Decoding Step},
  author={Yang, Jingyi and Chen, Guanxu and Hu, Xuhao and Shao, Jing},
  journal={arXiv preprint arXiv:2509.23924},
  year={2025}
}
```


## 🙏 Acknowledgments

We thank the open-source community for their wonderful work and valuable contributions:
- Models Source: [GSAI-ML](https://huggingface.co/GSAI-ML), [inclusionAI](https://huggingface.co/inclusionAI), [Dream-org](https://huggingface.co/Dream-org), [JetLM](https://huggingface.co/JetLM), [huggingface](https://huggingface.co/) for model supply or hosting
- Algorithm: [d1](https://github.com/dllm-reasoning/d1), [CJ-GRPO](https://github.com/yjyddq/EOSER-ASS-RL), [MDPO](https://github.com/autonomousvision/mdpo), [Coupled-GRPO](https://github.com/apple/ml-diffucoder), [SPG](https://github.com/facebookresearch/SPG), [BGPO](https://github.com/THU-KEG/BGPO), [VRPO](https://arxiv.org/pdf/2505.19223), [EBPO](https://arxiv.org/pdf/2602.08676), [d-TreeRPO](https://github.com/THU-BPM/d-TreeRPO)
- Training Framework: [verl](https://github.com/volcengine/verl), [BGPO](https://github.com/THU-KEG/BGPO), [Dream](https://github.com/DreamLM/Dream), [dFactory](https://github.com/inclusionAI/dFactory)
- Inference Acceleration/Engine: [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM), [lmdeploy](https://github.com/InternLM/lmdeploy), [SGLang](https://github.com/sgl-project/sglang)
- Evaluation Framework: [opencompass](https://github.com/open-compass/opencompass), [LLaDA](https://github.com/ML-GSAI/LLaDA)
- Environment Dependencies: [verl](https://github.com/volcengine/verl), [opencompass](https://github.com/open-compass/opencompass), [flash-attn](https://flashattn.dev/)
