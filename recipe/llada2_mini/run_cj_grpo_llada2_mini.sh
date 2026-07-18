#!/bin/bash
set -euo pipefail
set -x

cleanup() {
    ray stop --force || true
    rm -rf /tmp/ray || true
}

while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    --model)
      model="$2"
      shift 2
      ;;
    --model_path)
      model_path="$2"
      shift 2
      ;;
    --task)
      task="$2"
      shift 2
      ;;
    --engine)
      engine="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_PROJECT="${WANDB_PROJECT:-DARE}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HOME="${HF_HOME:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-0}"

unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF

model="${model:-llada2}"
model_path="${model_path:-models/LLaDA2.0-mini}"
task="${task:-math}"
algorithm="cj-grpo"
engine="${engine:-sglang}"

valid_tasks=("math" "code" "sudoku" "countdown")
if [[ ! " ${valid_tasks[*]} " =~ " ${task} " ]]; then
    echo "Error: Invalid task '${task}'"
    echo "Supported tasks: ${valid_tasks[*]}"
    exit 1
fi

valid_models=("llada2")
if [[ ! " ${valid_models[*]} " =~ " ${model} " ]]; then
    echo "Error: Invalid model '${model}'"
    echo "Supported models: ${valid_models[*]}"
    exit 1
fi

valid_engines=("sglang")
if [[ ! " ${valid_engines[*]} " =~ " ${engine} " ]]; then
    echo "Error: Invalid engine '${engine}'"
    echo "Supported engines: ${valid_engines[*]}"
    exit 1
fi

echo "[INFO] Cleaning up old Ray..."
cleanup

n_gpus_per_node=$(echo "$CUDA_VISIBLE_DEVICES" | tr "," "\n" | wc -l)
num_cpus=$(nproc --all 2>/dev/null || echo 128)

echo "[INFO] Starting Ray head..."
ray start --head \
  --node-ip-address=127.0.0.1 \
  --port=6379 \
  --num-gpus="${n_gpus_per_node}" \
  --num-cpus="${num_cpus}"

baseline="${model}-${task}-${algorithm}-${engine}"

if [ "$task" == "math" ]; then
    train_files="['data/preprocessed/rl/train/math_1.parquet','data/preprocessed/rl/train/gsm8k_1.parquet']"
    val_files="['data/preprocessed/rl/test/math500_1.parquet','data/preprocessed/rl/test/gsm8k_1.parquet']"
    max_prompt_length=512
    max_response_length=2048
    total_epoch=1
elif [ "$task" == "code" ]; then
    train_files="['data/preprocessed/rl/train/lcbv5-K8_1.parquet','data/preprocessed/rl/train/primeintellect-K8_1.parquet','data/preprocessed/rl/train/taco-K8_1.parquet']"
    val_files="['data/preprocessed/rl/test/mbpp_1.parquet','data/preprocessed/rl/test/humaneval_1.parquet','data/preprocessed/rl/test/humanevalplus_1.parquet']"
    max_prompt_length=1024
    max_response_length=2048
    total_epoch=5
elif [ "$task" == "countdown" ]; then
    train_files="['data/preprocessed/rl/train/countdown-n20000_1.parquet']"
    val_files="['data/preprocessed/rl/test/countdown_1.parquet']"
    max_prompt_length=512
    max_response_length=2048
    total_epoch=1
else
    train_files="['data/preprocessed/rl/train/sudoku-n20000_1.parquet']"
    val_files="['data/preprocessed/rl/test/sudoku_1.parquet']"
    max_prompt_length=512
    max_response_length=2048
    total_epoch=1
fi

mask_token_id=156895
pad_token_id=156892
batch_size=16
n_rollout=8
lr=5e-7
ppo_micro_batch_size_per_gpu=1
train_temperature=0.6
block_length=32
mc_num=1
n_l=1
fsdp_size=-1
rollout_gpu_memory_utilization=0.5
rollout_mem_fraction_static=0.5
rollout_max_running_requests=4
rollout_attention_backend="flashinfer"
rollout_dllm_algorithm="LowConfidence"
rollout_dllm_algorithm_config="recipe/llada2_mini/sglang_low_confidence_llada2.yaml"

timestamp=$(date +"%Y%m%d_%H%M%S")
project_name=$WANDB_PROJECT
exp_name="${baseline}-bsz${batch_size}-n${n_rollout}-prompt${max_prompt_length}-response${max_response_length}-lr${lr}-temp${train_temperature}-gpu${n_gpus_per_node}-${timestamp}"
ckpt_dir=./ckpt/${project_name}/${exp_name}
log_dir=./logs/${project_name}/${exp_name}
mkdir -p "${ckpt_dir}"
mkdir -p "${log_dir}"

python3 -m verl.trainer.dllm_main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.name=${algorithm} \
    reward_model.reward_manager=dllm \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward_model.reward_kwargs.max_resp_len=$max_response_length \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=$batch_size \
    data.val_batch_size=64 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation="error" \
    data.trust_remote_code=True \
    data.return_full_prompt=True \
    +actor_rollout_ref.algorithm.name=${algorithm} \
    +actor_rollout_ref.model.name=${model} \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.attn_implementation="sdpa" \
    +actor_rollout_ref.model.baseline=$baseline \
    +actor_rollout_ref.model.override_config.mask_token_id=$mask_token_id \
    +actor_rollout_ref.model.override_config.pad_token_id=$pad_token_id \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5120 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=bfloat16 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=10000 \
    +actor_rollout_ref.actor.mc_num=$mc_num \
    +actor_rollout_ref.actor.n_l=$n_l \
    +actor_rollout_ref.actor.block_length=$block_length \
    +actor_rollout_ref.actor.cfg_scale=0.0 \
    +actor_rollout_ref.actor.baseline=$baseline \
    actor_rollout_ref.rollout.name=$engine \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref.rollout.use_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$n_rollout \
    actor_rollout_ref.rollout.temperature=$train_temperature \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=50 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=11000 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    +actor_rollout_ref.rollout.mem_fraction_static=$rollout_mem_fraction_static \
    +actor_rollout_ref.rollout.max_running_requests=$rollout_max_running_requests \
    +actor_rollout_ref.rollout.attention_backend=$rollout_attention_backend \
    +actor_rollout_ref.rollout.dllm_algorithm=$rollout_dllm_algorithm \
    +actor_rollout_ref.rollout.dllm_algorithm_config=$rollout_dllm_algorithm_config \
    +actor_rollout_ref.rollout.return_cj_trajectory=True \
    +actor_rollout_ref.rollout.block_length=$block_length \
    +actor_rollout_ref.rollout.mask_token_id=$mask_token_id \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params=10000 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=["console","wandb"] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.val_before_train=True \
    +trainer.val_only=False \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.default_local_dir=$ckpt_dir \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=$total_epoch \
    custom_reward_function.path="verl/utils/reward_score/__init__.py" \
    custom_reward_function.name="dllm_rm" \
    >> "${log_dir}/${baseline}-${timestamp}.out" \
    2>> "${log_dir}/${baseline}-${timestamp}.err" &
