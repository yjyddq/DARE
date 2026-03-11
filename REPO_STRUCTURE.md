# DARE 仓库组织架构

> **DARE** = **d**LLM **A**lignment and **R**einforcement **E**xecutor
>
> 一个专为扩散大语言模型 (dLLMs) 设计的监督微调 (SFT) 和强化学习 (RL) 训练及评估框架。

---

## 📁 顶层目录总览

```
DARE/
├── assets/             # 项目图片资源（Logo、架构图等）
├── data/               # 数据集（预处理后的训练/测试数据）
├── models/             # 各 dLLM 模型的配置与建模文件
├── opencompass/         # 评估框架（基于 OpenCompass）
├── recipe/             # 训练脚本配方（按模型分类的一键训练脚本）
├── scripts/            # 通用脚本（评估、数据预处理、检查点转换等）
├── verl/               # 核心训练框架（基于 verl，RL/SFT 的核心实现）
├── .gitignore
├── LICENSE             # Apache 2.0 许可证
├── README.md           # 项目主文档
└── requirements.txt    # Python 依赖列表
```

---

## 🔍 各模块详细说明

### 1. `assets/` — 静态资源

存放项目文档使用的图片资源。

```
assets/
├── DARE_logo.png                # 项目 Logo
├── optimization_plan_bdlm.png   # BDLM（块扩散模型）优化方案架构图
└── optimization_plan_mdlm.png   # MDLM（掩码扩散模型）优化方案架构图
```

---

### 2. `data/` — 数据集

存放预处理好的训练与测试数据集，按任务类型组织。

```
data/
└── preprocessed/
    ├── rl/          # 强化学习数据
    │   ├── train/   # 训练集（.parquet 格式）
    │   └── test/    # 测试集（.parquet 格式）
    └── sft/         # 监督微调数据
        ├── train/   # 训练集（.parquet 格式）
        └── test/    # 测试集（.parquet 格式）
```

数据预处理脚本位于 `verl/utils/preprocess/` 和 `scripts/` 目录下。

---

### 3. `models/` — 模型定义文件

包含各 dLLM 模型的 HuggingFace 兼容配置、建模代码和 tokenizer 配置。用户需要将下载的模型权重（`.safetensors`）放置到对应目录中。

```
models/
├── LLaDA-8B-Instruct/        # LLaDA 8B 指令微调模型
│   ├── config.json
│   ├── configuration_llada.py  # 模型配置类
│   ├── modeling_llada.py       # 模型实现（含注意力后端优化）
│   ├── tokenizer.json
│   └── ...
├── Dream-v0-Instruct-7B/     # Dream 7B 指令微调模型
│   ├── configuration_dream.py
│   ├── modeling_dream.py
│   ├── generation_utils.py     # 生成工具
│   ├── generation_utils_block.py
│   └── ...
├── SDAR-1dot7B-Chat/          # SDAR 1.7B 对话模型
├── SDAR-4B-Chat/              # SDAR 4B 对话模型
├── SDAR-8B-Chat/              # SDAR 8B 对话模型
│   ├── configuration_sdar.py
│   ├── modeling_sdar.py
│   ├── fused_linear_diffusion_cross_entropy.py  # 融合线性交叉熵（节省显存）
│   └── ...
├── SDAR-30B-A3B-Chat/         # SDAR 30B-A3B 对话模型
├── LLaDA2.0-mini/             # LLaDA 2.0 mini（MoE 架构）
└── LLaDA2.1-mini/             # LLaDA 2.1 mini（MoE 架构）
```

---

### 4. `verl/` — 核心训练框架

这是 DARE 项目的核心，基于 [verl](https://github.com/volcengine/verl) 框架进行扩展，实现了 dLLM 的 SFT、RL 和 DPO 训练。

```
verl/
├── __init__.py
├── protocol.py               # 数据协议定义
├── models/                   # 模型注册与加载
├── trainer/                  # 训练器（SFT/RL/DPO 的训练循环）
├── workers/                  # 分布式训练工作器
├── utils/                    # 工具函数集合
├── single_controller/        # 单控制器（Ray 分布式调度）
├── third_party/              # 第三方集成（SGLang、vLLM）
├── tools/                    # 工具调用支持
└── version/                  # 版本信息
```

#### 4.1 `verl/models/` — 模型注册与权重加载

```
verl/models/
├── __init__.py
├── registry.py               # 模型注册表
├── weight_loader_registry.py  # 权重加载器注册
├── llama/                    # LLaMA 系列模型适配
├── qwen2/                    # Qwen2 系列模型适配
├── mcore/                    # Megatron-Core 集成
└── transformers/             # HuggingFace Transformers 适配
```

#### 4.2 `verl/trainer/` — 训练器

训练的入口和主循环实现。

```
verl/trainer/
├── config/                         # Hydra 配置文件
│   ├── sft_trainer.yaml            # SFT 训练配置
│   ├── ppo_trainer.yaml            # PPO/RL 训练配置
│   ├── dpo_trainer.yaml            # DPO 训练配置
│   ├── evaluation.yaml             # 评估配置
│   ├── generation.yaml             # 生成配置
│   └── ppo_megatron_trainer.yaml   # Megatron PPO 配置
│
├── # === SFT 训练器（监督微调）===
├── fsdp_sft_trainer.py             # 通用 FSDP SFT 训练器
├── llada_fsdp_sft_trainer.py       # LLaDA 专用 SFT 训练器
├── dream_fsdp_sft_trainer.py       # Dream 专用 SFT 训练器
├── sdar_fsdp_sft_trainer.py        # SDAR 专用 SFT 训练器
│
├── # === RL 训练器（强化学习）===
├── dllm_main_ppo.py                # dLLM RL 训练入口
├── main_ppo.py                     # 通用 PPO 训练入口
│
├── ppo/                            # PPO/RL 核心算法
│   ├── core_algos.py               # 通用 RL 核心算法
│   ├── dllm_core_algos.py          # dLLM 专用 RL 核心算法（d1, coupled-grpo, cj-grpo, spg, bgpo 等）
│   ├── mdpo_algos.py               # MDPO 算法实现
│   ├── ray_trainer.py              # 通用 Ray 分布式训练器
│   ├── dllm_ray_trainer.py         # dLLM Ray 分布式训练器
│   ├── reward.py                   # 通用奖励函数
│   ├── dllm_reward.py              # dLLM 奖励函数
│   ├── metric_utils.py             # 评估指标工具
│   └── dllm_metric_utils.py        # dLLM 评估指标工具
│
├── # === DPO/VRPO 训练器（偏好优化）===
├── dllm_main_dpo.py                # dLLM DPO 训练入口
└── dpo/
    └── llada_fsdp_dpo_trainer.py   # LLaDA FSDP DPO 训练器
```

#### 4.3 `verl/workers/` — 分布式工作器

实现了分布式训练中各个角色（Actor、Critic、Rollout 等）的具体逻辑。

```
verl/workers/
├── fsdp_workers.py              # 通用 FSDP Worker
├── dllm_fsdp_workers.py         # dLLM 专用 FSDP Worker
├── megatron_workers.py          # Megatron Worker
│
├── actor/                       # Actor（策略网络）
│   ├── base.py                  # Actor 基类
│   ├── dp_actor.py              # 通用数据并行 Actor
│   ├── llada_dp_actor_d1.py     # LLaDA + d1 算法
│   ├── llada_dp_actor_coupled_grpo.py  # LLaDA + Coupled-GRPO
│   ├── llada_dp_actor_cj_grpo.py      # LLaDA + CJ-GRPO
│   ├── llada_dp_actor_spg.py          # LLaDA + SPG
│   ├── llada_dp_actor_bgpo.py         # LLaDA + BGPO
│   ├── llada_dp_actor_vrpo.py         # LLaDA + VRPO
│   ├── llada_dp_actor_mdpo.py         # LLaDA + MDPO
│   ├── dream_dp_actor_d1.py           # Dream + d1
│   ├── dream_dp_actor_coupled_grpo.py # Dream + Coupled-GRPO
│   ├── dream_dp_actor_cj_grpo.py      # Dream + CJ-GRPO
│   ├── dream_dp_actor_spg.py          # Dream + SPG
│   ├── dream_dp_actor_bgpo.py         # Dream + BGPO
│   ├── dream_dp_actor_vrpo.py         # Dream + VRPO
│   ├── dream_dp_actor_mdpo.py         # Dream + MDPO
│   ├── sdar_dp_actor_bgpo.py          # SDAR + BGPO
│   └── megatron_actor.py              # Megatron Actor
│
├── critic/                      # Critic（价值网络）
│   ├── base.py
│   ├── dp_critic.py             # 数据并行 Critic
│   └── megatron_critic.py       # Megatron Critic
│
├── rollout/                     # Rollout（推理/采样）
│   ├── base.py
│   ├── # --- LLaDA Rollout ---
│   ├── llada_rollout.py         # LLaDA 标准 Rollout
│   ├── fast_llada_rollout.py    # LLaDA 快速 Rollout（Fast-dLLM 加速）
│   ├── cj_llada_rollout.py      # LLaDA CJ-GRPO Rollout
│   ├── fast_cj_llada_rollout.py # LLaDA 快速 CJ-GRPO Rollout
│   ├── mdpo_llada_rollout.py    # LLaDA MDPO Rollout
│   ├── # --- Dream Rollout ---
│   ├── dream_rollout.py         # Dream 标准 Rollout
│   ├── fast_dream_rollout.py    # Dream 快速 Rollout（Fast-dLLM 加速）
│   ├── cj_dream_rollout.py      # Dream CJ-GRPO Rollout
│   ├── fast_cj_dream_rollout.py # Dream 快速 CJ-GRPO Rollout
│   ├── mdpo_dream_rollout.py    # Dream MDPO Rollout
│   ├── # --- 通用 Rollout ---
│   ├── hf_rollout.py            # HuggingFace 后端 Rollout
│   ├── generate.py              # 通用生成逻辑
│   ├── generation_utils.py      # 生成工具函数
│   ├── generation_utils_block.py # 块扩散生成工具
│   ├── rollout_utils.py         # Rollout 辅助工具
│   ├── schemas.py               # 数据 Schema
│   ├── tokenizer.py             # Tokenizer 包装
│   ├── async_server.py          # 异步推理服务
│   ├── # --- 推理引擎集成 ---
│   ├── lmdeploy_rollout/        # lmdeploy 加速后端（用于 SDAR）
│   │   └── lmdeploy_rollout_server.py
│   ├── sglang_rollout/          # SGLang 加速后端
│   │   ├── sglang_rollout.py
│   │   ├── sglang_sdar_rollout.py
│   │   ├── async_sglang_server.py
│   │   └── utils.py
│   ├── vllm_rollout/            # vLLM 后端
│   └── naive/                   # 朴素实现
│
├── reward_manager/              # 奖励管理器
│   ├── batch.py                 # 批量奖励计算
│   ├── naive.py                 # 朴素奖励管理
│   ├── dllm.py                  # dLLM 专用奖励管理
│   ├── dapo.py                  # DAPO 奖励管理
│   └── prime.py                 # PRIME 奖励管理
│
├── reward_model/                # 奖励模型
│   ├── base.py
│   └── megatron/
│
└── sharding_manager/            # 分片管理器（分布式权重分片策略）
    ├── base.py
    ├── fsdp_ulysses.py          # FSDP + Ulysses 序列并行
    ├── fsdp_vllm.py             # FSDP + vLLM
    ├── fsdp_sglang.py           # FSDP + SGLang
    ├── fsdp_sglang_sdar.py      # FSDP + SGLang（SDAR 专用）
    ├── fsdp_lmdeploy_server.py  # FSDP + lmdeploy
    ├── megatron_vllm.py         # Megatron + vLLM
    └── megatron_sglang.py       # Megatron + SGLang
```

#### 4.4 `verl/utils/` — 工具函数

```
verl/utils/
├── # --- 数据处理 ---
├── dataset/                     # 数据集加载器
│   ├── rl_dataset.py            # RL 数据集
│   ├── llada_sft_dataset.py     # LLaDA SFT 数据集
│   ├── dream_sft_dataset.py     # Dream SFT 数据集
│   ├── sdar_sft_dataset.py      # SDAR SFT 数据集
│   ├── sft_dataset.py           # 通用 SFT 数据集
│   └── rm_dataset.py            # 奖励模型数据集
├── preprocess/                  # 数据预处理脚本
│   ├── preprocess.py            # RL 数据预处理
│   ├── preprocess_sft.py        # SFT 数据预处理
│   ├── preprocess_dpo.py        # DPO 数据预处理
│   └── preprocess_sudoku_countdown.py  # Sudoku/Countdown 任务预处理
│
├── # --- 奖励评分 ---
├── reward_score/                # 奖励评分函数
│   ├── math.py / math_verify.py # 数学题奖励评分
│   ├── code_reward.py           # 代码题奖励评分
│   ├── gsm8k.py                 # GSM8K 评分
│   └── ...
│
├── # --- 分布式与系统 ---
├── distributed.py               # 分布式训练工具
├── fsdp_utils.py                # FSDP 工具函数
├── ulysses.py                   # Ulysses 序列并行
├── ray_utils.py                 # Ray 分布式工具
├── device.py                    # 设备管理
├── net_utils.py                 # 网络工具
├── rendezvous/                  # 分布式 Rendezvous
│
├── # --- 模型与检查点 ---
├── model.py                     # 模型工具
├── checkpoint/                  # 检查点管理
├── convert_ckpt_to_hf.py        # FSDP 检查点转 HuggingFace 格式
├── tokenizer.py                 # Tokenizer 工具
│
├── # --- 监控与日志 ---
├── tracking.py                  # 实验追踪（WandB/SwanLab 集成）
├── logger/                      # 日志系统
├── logging_utils.py             # 日志工具
├── metric/                      # 指标计算
├── flops_counter.py             # FLOPs 计数器
│
├── # --- 其他工具 ---
├── config.py                    # 配置管理
├── memory_buffer.py             # 显存缓冲区
├── seqlen_balancing.py          # 序列长度均衡
├── torch_functional.py          # PyTorch 工具函数
├── torch_dtypes.py              # 数据类型工具
└── import_utils.py              # 动态导入工具
```

#### 4.5 `verl/single_controller/` — 分布式调度

```
verl/single_controller/
├── base/                        # 基础调度类
└── ray/                         # Ray 分布式调度实现
```

#### 4.6 `verl/third_party/` — 第三方集成

```
verl/third_party/
├── sglang/                      # SGLang 推理引擎集成
└── vllm/                        # vLLM 推理引擎集成
```

#### 4.7 `verl/tools/` — 工具调用

```
verl/tools/
├── base_tool.py                 # 工具基类
├── gsm8k_tool.py                # GSM8K 数学工具
├── sandbox_fusion_tools.py      # 沙箱融合工具
├── search_tool.py               # 搜索工具
├── schemas.py                   # Schema 定义
└── utils/                       # 工具辅助函数
```

---

### 5. `opencompass/` — 评估框架

基于 [OpenCompass](https://github.com/open-compass/opencompass) 的评估框架，专门为 dLLM 模型集成了自定义评估能力。

```
opencompass/
├── setup.py                     # 安装脚本
├── run.py                       # 评估入口脚本
├── requirements.txt             # 评估环境依赖
├── README.md                    # 评估使用指南
├── dataset-index.yml            # 数据集索引
│
├── opencompass/                  # OpenCompass 核心代码
│   ├── models/                  # 模型适配
│   │   ├── llada.py             # LLaDA 模型适配
│   │   ├── llada2.py            # LLaDA2.0 适配
│   │   ├── llada2dot1.py        # LLaDA2.1 适配
│   │   ├── llada_moe.py         # LLaDA MoE 适配
│   │   ├── dream.py             # Dream 模型适配
│   │   ├── sdar.py              # SDAR 模型适配
│   │   ├── sdar_generate.py     # SDAR 生成逻辑
│   │   ├── sglang_model.py      # SGLang 后端适配
│   │   ├── turbomind.py         # TurboMind (lmdeploy) 适配
│   │   └── ...                  # 其他模型（HuggingFace、API 等）
│   ├── configs/models/dllm/     # dLLM 模型评估配置
│   │   ├── llada_instruct_8b.py
│   │   ├── dream_v0_instruct_7b.py
│   │   ├── sdar_8b_chat.py
│   │   ├── sglang_*.py          # SGLang 后端配置
│   │   ├── lmdeploy_*.py        # lmdeploy 后端配置
│   │   └── ...
│   ├── datasets/                # 评测数据集配置
│   ├── runners/                 # 任务运行器
│   ├── tasks/                   # 评测任务
│   ├── summarizers/             # 结果汇总
│   └── metrics/                 # 评测指标
│
├── # --- 评估示例脚本 ---
├── llada_examples/              # LLaDA 评估示例
├── dream_examples/              # Dream 评估示例
├── sdar_examples/               # SDAR 评估示例
├── llada2_mini_examples/        # LLaDA2.0-mini 评估示例
├── llada2dot1_mini_examples/    # LLaDA2.1-mini 评估示例
├── llada_moe_7b_a1b_examples/   # LLaDA MoE 评估示例
├── sdar_30b_a3b_examples/       # SDAR-30B 评估示例
│
├── docs/                        # 文档
├── tests/                       # 测试
└── tools/                       # 评估辅助工具
```

---

### 6. `recipe/` — 训练配方脚本

按模型类型组织的一键训练脚本。每个脚本封装了完整的训练命令和超参数配置。

```
recipe/
├── llada/                               # LLaDA 模型训练脚本
│   ├── run_sft_llada_8b_instruct.sh     # SFT 全参微调
│   ├── run_sft_peft_llada_8b_instruct.sh # SFT PEFT (LoRA) 微调
│   ├── run_d1_llada_8b_instruct.sh      # d1 算法 RL 训练
│   ├── run_coupled_grpo_llada_8b_instruct.sh  # Coupled-GRPO
│   ├── run_cj_grpo_llada_8b_instruct.sh       # CJ-GRPO
│   ├── run_spg_llada_8b_instruct.sh           # SPG
│   ├── run_bgpo_llada_8b_instruct.sh          # BGPO
│   ├── run_vrpo_llada_8b_instruct.sh          # VRPO（偏好优化）
│   └── run_mdpo_llada_8b_instruct.sh          # MDPO
│
├── dream/                               # Dream 模型训练脚本
│   ├── run_sft_dream_7b_instruct.sh
│   ├── run_sft_peft_dream_7b_instruct.sh
│   ├── run_d1_dream_7b_instruct.sh
│   ├── run_coupled_grpo_dream_7b_instruct.sh
│   ├── run_cj_grpo_dream_7b_instruct.sh
│   ├── run_spg_dream_7b_instruct.sh
│   ├── run_bgpo_dream_7b_instruct.sh
│   ├── run_vrpo_dream_7b_instruct.sh
│   └── run_mdpo_dream_7b_instruct.sh
│
└── sdar/                                # SDAR 模型训练脚本
    ├── run_sft_peft_sdar_1dot7b_chat.sh
    ├── run_sft_peft_sdar_4b_chat.sh
    ├── run_sft_peft_sdar_8b_chat.sh
    ├── run_bgpo_sdar_1dot7b_chat.sh
    ├── run_bgpo_sdar_4b_chat.sh
    └── run_bgpo_sdar_8b_chat.sh
```

---

### 7. `scripts/` — 通用脚本

```
scripts/
├── # --- 评估脚本 ---
├── eval_llada1dot5.sh             # LLaDA 1.5 评估
├── eval_llada2_mini.sh            # LLaDA2.0-mini 评估
├── eval_llada2dot1_mini.sh        # LLaDA2.1-mini 评估
├── eval_llada_moe_7b_a1b.sh       # LLaDA MoE 评估
├── eval_sglang_llada2_mini.sh     # SGLang 后端 LLaDA2 评估
├── eval_sdar_1dot7b_chat.sh       # SDAR 各尺寸评估
├── eval_sdar_4b_chat.sh
├── eval_sdar_8b_chat.sh
├── eval_sdar_30b_a3b_chat.sh
├── eval_local_bench.sh            # 本地基准评估
│
├── # --- 数据处理 ---
├── preprocess_dataset.sh          # 数据集预处理
├── preprocess_dpo_dataset.sh      # DPO 数据集预处理
│
└── # --- 检查点转换 ---
    convert_ckpt_to_hf.sh         # FSDP 分片检查点 → HuggingFace 格式
```

---

## 🏗️ 架构设计

### 整体架构

```
┌──────────────────────────────────────────────────────────┐
│                     DARE 框架                             │
├──────────────────────────┬───────────────────────────────┤
│      训练子系统 (verl/)   │      评估子系统 (opencompass/)  │
│                          │                               │
│  ┌────────────────────┐  │  ┌─────────────────────────┐  │
│  │  Trainer 训练器      │  │  │  Models 模型适配         │  │
│  │  (SFT/RL/DPO)      │  │  │  (LLaDA/Dream/SDAR)    │  │
│  └────────┬───────────┘  │  ├─────────────────────────┤  │
│           │              │  │  Datasets 数据集配置      │  │
│  ┌────────▼───────────┐  │  ├─────────────────────────┤  │
│  │  Workers 工作器      │  │  │  Runners/Tasks 评测运行  │  │
│  │  Actor | Critic    │  │  ├─────────────────────────┤  │
│  │  Rollout | Reward  │  │  │  Summarizers 结果汇总    │  │
│  └────────┬───────────┘  │  └─────────────────────────┘  │
│           │              │                               │
│  ┌────────▼───────────┐  │                               │
│  │  推理加速引擎        │  │                               │
│  │  Fast-dLLM         │  │                               │
│  │  lmdeploy | SGLang │  │                               │
│  └────────────────────┘  │                               │
├──────────────────────────┴───────────────────────────────┤
│                models/ 模型定义文件                        │
│              data/ 预处理数据集                            │
│           recipe/ & scripts/ 训练与评估脚本                │
└──────────────────────────────────────────────────────────┘
```

### 模型类型

DARE 支持两大类扩散语言模型：

| 类型 | 全称 | 代表模型 | 推理加速 |
|------|------|---------|---------|
| **MDLM** | Masked Diffusion Language Model | LLaDA, Dream | Fast-dLLM (Block Cache) |
| **BDLM** | Block Diffusion Language Model | SDAR, LLaDA2.0 | lmdeploy, SGLang |

### RL 算法支持矩阵

| 算法 | LLaDA | Dream | SDAR |
|------|-------|-------|------|
| d1 | ✅ | ✅ | ❌ |
| Coupled-GRPO | ✅ | ✅ | ❌ |
| CJ-GRPO | ✅ | ✅ | ❌ |
| SPG | ✅ | ✅ | ❌ |
| BGPO | ✅ | ✅ | ✅ |
| VRPO | ✅ | ✅ | ❌ |
| MDPO | ✅ | ✅ | ❌ |

### 训练流程

```
数据预处理 (verl/utils/preprocess/)
    │
    ▼
SFT 训练 (verl/trainer/*_fsdp_sft_trainer.py)
    │
    ▼
RL 训练 (verl/trainer/dllm_main_ppo.py)
    │
    ├── Actor (verl/workers/actor/)  ──  策略网络，执行各 RL 算法
    ├── Critic (verl/workers/critic/)  ──  价值估计（可选）
    ├── Rollout (verl/workers/rollout/)  ──  采样生成
    │     ├── Fast-dLLM (MDLM 加速)
    │     ├── lmdeploy (BDLM 加速)
    │     └── SGLang (通用加速)
    └── Reward (verl/workers/reward_manager/)  ──  奖励计算
          └── reward_score/ (数学/代码/QA 评分)
    │
    ▼
检查点转换 (scripts/convert_ckpt_to_hf.sh)
    │
    ▼
评估 (opencompass/)
```

---

## 🔑 关键设计特点

1. **双环境隔离**：训练 (verl) 和评估 (opencompass) 使用独立的 Python 虚拟环境，避免依赖冲突。

2. **模型-算法解耦**：Actor 按 `{模型}_{算法}` 组合命名（如 `llada_dp_actor_d1.py`），便于独立扩展模型或算法。

3. **多后端推理加速**：Rollout 模块支持多种推理后端（HuggingFace、Fast-dLLM、lmdeploy、SGLang），通过 Sharding Manager 管理不同后端的权重分片策略。

4. **FSDP 分布式训练**：核心采用 PyTorch FSDP 实现分布式训练，可选 Megatron-Core 集成。

5. **Hydra 配置管理**：使用 Hydra 框架管理训练配置，配置文件位于 `verl/trainer/config/`。

6. **Ray 分布式调度**：使用 Ray 框架进行分布式任务调度，协调 Actor、Critic、Rollout 等工作器。
