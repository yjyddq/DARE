set -x
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Add memory fragmentation optimization
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="DARE"
export WANDB_API_KEY=42598cc56636f040038970a197ecd2c231a697cc
export WANDB_RESUME="allow"
export WANDB_MODE="offline"
export HF_HOME=/mnt/shared-storage-user/yangjingyi/huggingface
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1

echo "Usage: run_sft_peft.sh <nproc_per_node> <model_path> [other_configs...]"

nproc_per_node=${1:-8}
MODEL_PATH=${2:-/mnt/shared-storage-user/yangjingyi/BGPO/models/Dream-v0-Instruct-7B}

PROJECT_NAME=$WANDB_PROJECT
EXP_NAME="gsm8k-sft-dream-7b-instruct"
CKPT_DIR=/mnt/shared-storage-user/ai4good1-share/yangjingyi/models/${PROJECT_NAME}/${EXP_NAME}
LOG_DIR=/mnt/shared-storage-user/yangjingyi/BGPO/logs/${PROJECT_NAME}/${EXP_NAME}
mkdir -p ${CKPT_DIR}
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.dllm_fsdp_sft_trainer \
    data.train_files=data/preprocessed/sft/train/gsm8k_train.parquet \
    data.val_files=data/preprocessed/sft/test/gsm8k_test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=4096 \
    +data.mask_token_id=151666 \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=${MODEL_PATH} \
    model.trust_remote_code=True \
    +model.attn_implementation="flash_attention_2" \
    +model.fsdp_config.model_dtype=float32 \
    +model.external_lib=transformers_modules.Dream-v0-Instruct-7B \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.logger=["console","wandb"] \
    trainer.total_epochs=20 \
    trainer.total_training_steps=10000 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=false \
    model.lora_rank=32\
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    >> ${LOG_DIR}/gsm8k-${TIMESTAMP}.out \
    2>> ${LOG_DIR}/gsm8k-${TIMESTAMP}.err &

    # Or you can do this:
    # model.target_modules=[q_proj,v_proj] \
