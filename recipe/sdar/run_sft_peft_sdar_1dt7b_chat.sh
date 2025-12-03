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
export OMP_NUM_THREADS=1

echo "Usage: run_sft_peft.sh <nproc_per_node> <model_path> [other_configs...]"

nproc_per_node=${1:-8}
model_path=${2:-models/SDAR-1.7B-Chat}

timestamp=$(date +"%Y%m%d_%H%M%S")
project_name=$WANDB_PROJECT
exp_name="gsm8k-sft-sdar-1.7b-chat"
ckpt_dir=./ckpts/${project_name}/${exp_name}
log_dir=./logs/${project_name}/${exp_name}
mkdir -p ${ckpt_dir}
mkdir -p ${log_dir}


torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.dllm_fsdp_sft_trainer \
    data.train_files=data/preprocessed/sft/train/gsm8k_train.parquet \
    data.val_files=data/preprocessed/sft/test/gsm8k_test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=4096 \
    +data.mask_token_id=151669 \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=4 \
    mode1l.partial_pretrain=${model_path} \
    model.trust_remote_code=True \
    +model.attn_implementation="flash_attention_2" \
    +model.fsdp_config.model_dtype=float32 \
    +model.external_lib=transformers_modules.SDAR-1.7B-Chat \
    trainer.default_local_dir=$ckpt_dir \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.logger=["console","wandb"] \
    trainer.total_epochs=50 \
    trainer.total_training_steps=1000 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=false \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    >> ${log_dir}/gsm8k-${timestamp}.out \
    2>> ${log_dir}/gsm8k-${timestamp}.err &

    # Or you can do this:
    # model.target_modules=[q_proj,v_proj] \

    # trainer.total_epochs=1 \