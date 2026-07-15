#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Add memory fragmentation optimization
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="DARE"
export WANDB_API_KEY=
export WANDB_RESUME="allow"
export WANDB_MODE="offline"
export HF_HOME=
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE=1

skip_ray_cleanup=false
for arg in "$@"; do
  if [[ "$arg" == "--skip_ray_cleanup" ]]; then
    skip_ray_cleanup=true
    break
  fi
done

if [[ "$skip_ray_cleanup" == "false" ]]; then
    echo "[INFO] Cleaning up old Ray..."
    ray stop --force || true
    rm -rf /tmp/ray || true
else
    echo "[INFO] Skipping Ray cleanup..."
fi

# arguments parsing
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model)
      model="$2"
      shift; shift
      ;;
    --model_path)
      model_path="$2"
      shift; shift
      ;;
    --task)
      task="$2"
      shift; shift
      ;;
    --algorithm)
      algorithm="$2"
      shift; shift
      ;;
    --engine)
      engine="$2"
      shift; shift
      ;;
    --resume_path)
      resume_path="$2"
      shift; shift
      ;;
    --ckpt_dir)
      ckpt_dir="$2"
      shift; shift
      ;;
    --experiment_suffix)
      experiment_suffix="$2"
      shift; shift
      ;;
    --mc_num)
      mc_num="$2"
      shift; shift
      ;;
    --n_l)
      n_l="$2"
      shift; shift
      ;;
    --num_iterations)
      num_iterations="$2"
      shift; shift
      ;;
    --beta)
      espo_beta="$2"
      shift; shift
      ;;
    --save_freq)
      save_freq="$2"
      shift; shift
      ;;
    --test_freq)
      test_freq="$2"
      shift; shift
      ;;
    --max_actor_ckpt_to_keep)
      max_actor_ckpt_to_keep="$2"
      shift; shift
      ;;
    --foreground)
      foreground=true
      shift
      ;;
    --val_before_train)
      val_before_train="$2"
      shift; shift
      ;;
    --val_only)
      val_only="$2"
      shift; shift
      ;;
    --use_cache)
      use_cache="$2"
      shift; shift
      ;;
    --skip_ray_cleanup)
      shift
      ;;
    *)
      shift
      ;;
  esac
done

algorithm=${algorithm:-espo}
model=${model:-llada}
model_path=${model_path:-models/LLaDA-8B-Instruct}
resume_path=${resume_path}
experiment_suffix=${experiment_suffix:-}
engine=${engine:-hf}
mc_num=${mc_num:-2}
n_l=${n_l:-1}
num_iterations=${num_iterations:-8}
save_freq=${save_freq:-100}
test_freq=${test_freq:-10}
max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep:-1}
foreground=${foreground:-false}
val_before_train=${val_before_train:-False}
val_only=${val_only:-False}
use_cache=${use_cache:-True}

if [[ "$n_l" != "1" ]]; then
    echo "Error: ESPO does not use n_l grouping; set --n_l 1"
    exit 1
fi
if ! [[ "$mc_num" =~ ^[1-9][0-9]*$ && "$num_iterations" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: mc_num and num_iterations must be positive integers"
    exit 1
fi

# validate task
valid_tasks=("math" "code" "sudoku" "countdown")
if [[ ! " ${valid_tasks[@]} " =~ " ${task} " ]]; then
    echo "Error: Invalid task '$task'"
    echo "Supported tasks: ${valid_tasks[*]}"
    exit 1
fi

# validate model
valid_models=("llada")
if [[ ! " ${valid_models[@]} " =~ " ${model} " ]]; then
    echo "Error: Invalid model '$model'"
    echo "Supported models: ${valid_models[*]}"
    exit 1
fi

# validate algorithm
valid_algorithms=("d1" "coupled-grpo" "mdpo" "cj-grpo" "spg" "bgpo" "espo")
if [[ ! " ${valid_algorithms[@]} " =~ " ${algorithm} " ]]; then
    echo "Error: Invalid algorithm '$algorithm'"
    echo "Supported algorithms: ${valid_algorithms[*]}"
    exit 1
fi

if [[ "$algorithm" != "espo" ]]; then
    echo "Error: This recipe is reserved for ESPO"
    exit 1
fi

# validate engine
valid_engines=("hf")
if [[ ! " ${valid_engines[@]} " =~ " ${engine} " ]]; then
    echo "Error: Invalid engine '$engine'"
    echo "Supported engines: ${valid_engines[*]}"
    exit 1
fi

baseline="${model}-${task}-${algorithm}-${engine}"

if [ $task == "math" ]; then
    train_files="['data/preprocessed/rl/train/math_1.parquet','data/preprocessed/rl/train/gsm8k_1.parquet']"
    val_files="['data/preprocessed/rl/test/math500_1.parquet','data/preprocessed/rl/test/gsm8k_1.parquet']"
    max_prompt_length=512
    max_response_length=512
    num_diffusion_steps=$((max_response_length / 2))
    total_epoch=1
    espo_beta=${espo_beta:-0.001}
    espo_clip_ratio=0.2
    n_rollout=16
elif [ $task == "code" ]; then
    train_files="['data/preprocessed/rl/train/lcbv5-K8_1.parquet','data/preprocessed/rl/train/primeintellect-K8_1.parquet','data/preprocessed/rl/train/taco-K8_1.parquet']"
    val_files="['data/preprocessed/rl/test/mbpp_1.parquet','data/preprocessed/rl/test/humaneval_1.parquet','data/preprocessed/rl/test/humanevalplus_1.parquet']"
    max_prompt_length=1024
    max_response_length=512
    num_diffusion_steps=$max_response_length
    total_epoch=5
    espo_beta=${espo_beta:-0.01}
    espo_clip_ratio=0.2
    n_rollout=10
elif [ $task == "countdown" ]; then
    train_files="['data/preprocessed/rl/train/countdown-n20000_1.parquet']"
    val_files="['data/preprocessed/rl/test/countdown_1.parquet']"
    max_prompt_length=512
    max_response_length=256
    num_diffusion_steps=$((max_response_length / 2))
    total_epoch=1
    espo_beta=${espo_beta:-0.003}
    espo_clip_ratio=0.0
    n_rollout=6
elif [ $task == "sudoku" ]; then
    train_files="['data/preprocessed/rl/train/sudoku-n20000_1.parquet']"
    val_files="['data/preprocessed/rl/test/sudoku_1.parquet']"
    max_prompt_length=512
    max_response_length=256
    num_diffusion_steps=$((max_response_length / 2))
    total_epoch=1
    espo_beta=${espo_beta:-0.01}
    espo_clip_ratio=0.2
    n_rollout=6
fi

# Set token IDs based on model
case $model in
    "llada")
        mask_token_id=126336
        pad_token_id=126081
        ;;
    "dream")
        mask_token_id=151666
        pad_token_id=151643
        ;;
    "sdar")
        mask_token_id=151669
        pad_token_id=151643
        ;;
    *)
        echo "Error: Unknown model '$model'"
        exit 1
        ;;
esac

# parameters setting
n_gpus_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
batch_size=16  # batch_size must be greater than the number of GPUs used
lr=5e-7
ppo_micro_batch_size_per_gpu=1  # gradient accumulation = batch_size / ppo_micro_batch_size_per_gpu
train_temperature=1.0
adv_estimator=rloo
actor_use_kl_loss=True
actor_kl_loss_coef=$espo_beta
actor_kl_loss_type=k2

# diffusion related parameters
val_num_diffusion_steps=$max_response_length
block_length=32
actor_block_length=$block_length

timestamp=$(date +"%Y%m%d_%H%M%S")
project_name=$WANDB_PROJECT
exp_name="${baseline}-bsz${batch_size}-n${n_rollout}-prompt${max_prompt_length}-response${max_response_length}-step${num_diffusion_steps}-lr${lr}-temp${train_temperature}-iter${num_iterations}-mc_num${mc_num}-beta${espo_beta}-gpu${n_gpus_per_node}-${timestamp}"
if [[ -n "$experiment_suffix" ]]; then
    if [[ ! "$experiment_suffix" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Error: experiment suffix contains unsupported characters: $experiment_suffix"
        exit 1
    fi
    exp_name="${exp_name}-${experiment_suffix}"
fi
ckpt_dir=./ckpts/${project_name}/${exp_name}
log_dir=./logs/${project_name}/${exp_name}
mkdir -p ${log_dir}

# Build resume args based on whether ckpt_dir is specified
if [ -n "$resume_path" ]; then
    resume_args="trainer.resume_mode=resume_path trainer.resume_from_path=$resume_path trainer.default_local_dir=$ckpt_dir"
    log_suffix=$(echo "$ckpt_dir" | awk -F'/' '{print $(NF-1)"-"$NF}')
else
    resume_args="trainer.default_local_dir=$ckpt_dir"
    mkdir -p ${ckpt_dir}
    log_suffix=$(echo "$model_path" | awk -F'/' '{print $(NF-1)"-"$NF}')
fi

run_training() {
python3 -m verl.trainer.dllm_main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    +algorithm.name=${algorithm} \
    reward_model.reward_manager=dllm \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward_model.reward_kwargs.max_resp_len=$max_response_length \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=$batch_size \
    data.val_batch_size=1 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation="error" \
    +actor_rollout_ref.algorithm.name=${algorithm} \
    +actor_rollout_ref.model.name=${model} \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$batch_size \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5120 \
    actor_rollout_ref.actor.use_kl_loss=$actor_use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$actor_kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$actor_kl_loss_type \
    actor_rollout_ref.actor.clip_ratio=$espo_clip_ratio \
    actor_rollout_ref.actor.clip_ratio_low=$espo_clip_ratio \
    actor_rollout_ref.actor.clip_ratio_high=$espo_clip_ratio \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    +actor_rollout_ref.actor.logp_estimation=elbo \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.attn_implementation="flash_attention_2" \
    +actor_rollout_ref.model.baseline=$baseline \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=bfloat16 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE:-2} \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LLaDALlamaBlock] \
    +actor_rollout_ref.actor.mc_num=$mc_num \
    +actor_rollout_ref.actor.n_l=$n_l \
    +actor_rollout_ref.actor.num_iterations=$num_iterations \
    +actor_rollout_ref.actor.cfg_scale=0.0 \
    +actor_rollout_ref.actor.espo_reduce_var=true \
    +actor_rollout_ref.actor.baseline=$baseline \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    +actor_rollout_ref.rollout.use_cache=$use_cache \
    +actor_rollout_ref.rollout.dual_cache=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$n_rollout \
    actor_rollout_ref.rollout.temperature=$train_temperature \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    +actor_rollout_ref.rollout.val_kwargs.num_diffusion_steps=$val_num_diffusion_steps \
    actor_rollout_ref.rollout.max_num_batched_tokens=11000 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    +actor_rollout_ref.rollout.num_diffusion_steps=$num_diffusion_steps \
    +actor_rollout_ref.rollout.block_length=$block_length \
    +actor_rollout_ref.rollout.mc_num=$mc_num \
    +actor_rollout_ref.rollout.n_l=$n_l \
    +actor_rollout_ref.rollout.cfg_scale=0.0 \
    +actor_rollout_ref.ref.mc_num=$mc_num \
    +actor_rollout_ref.ref.n_l=$n_l \
    +actor_rollout_ref.ref.num_iterations=$num_iterations \
    +actor_rollout_ref.ref.cfg_scale=0.0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LLaDALlamaBlock] \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=["console","wandb"] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.val_before_train=$val_before_train \
    +trainer.val_only=$val_only \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    $resume_args \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.max_actor_ckpt_to_keep=$max_actor_ckpt_to_keep \
    trainer.total_epochs=$total_epoch \
    custom_reward_function.path="verl/utils/reward_score/__init__.py" \
    custom_reward_function.name="dllm_rm"
}

if [[ "$foreground" == "true" ]]; then
    run_training >> ${log_dir}/${baseline}-${timestamp}.out \
        2>> ${log_dir}/${baseline}-${timestamp}.err
else
    run_training >> ${log_dir}/${baseline}-${timestamp}.out \
        2>> ${log_dir}/${baseline}-${timestamp}.err &
fi

# reward_model.reward_manager=dllm: used to select reward_manager in dllm_reward.load_reward_manager()
# llada does not support gradient_checkpointing
# custom_reward_function.name: stored as self.reward_fn, will be called using compute_reward() in ray_trainer
