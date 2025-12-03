#!/bin/bash
set -e

export TORCHDYNAMO_DISABLE=1
export HF_HOME=
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HUB_OFFLINE=1
export COMPASS_DATA_CACHE=opencompass
cd opencompass

# parameter parsing
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --task)
      task="$2"
      shift; shift
      ;;
    --model)
      model="$2"
      shift; shift
      ;;
    --engine)
      engine="$2"
      shift; shift
      ;;
    *)
      shift
      ;;
  esac
done

model=${model:-Dream-v0-Instruct-7B}
engine=${engine:-hf}

if [ -z "${task}" ]; then
  echo "Usage: bash eval_dream.sh ${task}"
  echo "Optional task: mmlu, mmlupro, hellaswag, arcc, gsm8k_confidence math_confidence gpqa_confidence humaneval_logits mbpp_confidence gsm8k_short math_short"
  exit 1
fi

timestamp=$(date +"%Y%m%d_%H%M%S")
exp_name="eval_${model}_${task}"
log_dir=logs/EVAL/${exp_name}
mkdir -p ${log_dir}

# task Execution Map
case "${task}" in
  mmlu)
    py_script=dream_examples/dream_instruct_gen_mmlu_length128.py
    work_dir=outputs/dream_instruct_mmlu_length128
    ;;
  mmlupro)
    py_script=dream_examples/dream_instruct_gen_mmlupro_length128.py
    work_dir=outputs/dream_instruct_mmlupro_length128
    ;;
  hellaswag)
    py_script=dream_examples/dream_instruct_gen_hellaswag_length3.py
    work_dir=outputs/dream_instruct_hellaswag_length3
    ;;
  arcc)
    py_script=dream_examples/dream_instruct_gen_arcc_length512.py
    work_dir=outputs/dream_instruct_arcc_length512
    ;;
  gpqa)
    py_script=dream_examples/dream_instruct_gen_gpqa_length128.py
    work_dir=outputs/dream_instruct_gen_gpqa_length128
    ;;
  humaneval)
    py_script=dream_examples/dream_instruct_gen_humaneval_length512.py
    work_dir=outputs/dream_instruct_gen_humaneval_length512
    ;;
  mbpp)
    py_script=dream_examples/dream_instruct_gen_mbpp_length512.py
    work_dir=outputs/dream_instruct_gen_mbpp_length512
    ;;
  gsm8k)
    py_script=dream_examples/dream_instruct_gen_gsm8k_length256.py
    work_dir=outputs/dream_instruct_gen_gsm8k_length256
    ;;
  math)
    py_script=dream_examples/dream_instruct_gen_math_length512.py
    work_dir=outputs/dream_instruct_gen_math_length512
    ;;
  olympiad)
    py_script=dream_examples/dream_instruct_gen_olympiadbench_length2048.py
    work_dir=outputs/dream_instruct_gen_olympiadbench_length2048
    ;;
  aime2024)
    py_script=dream_examples/dream_instruct_gen_aime2024_length2048.py
    work_dir=outputs/dream_instruct_gen_aime2024_length2048
    ;;
  aime2025)
    py_script=dream_examples/dream_instruct_gen_aime2025_length2048.py
    work_dir=outputs/dream_instruct_gen_aime2025_length2048
    ;;
  *)
    echo "Unknown task: ${task}"
    exit 1
    ;;
esac

echo "task: ${task}"
echo "model: ${model}"
echo "Script: ${py_script}"
echo "Work Dir: ${work_dir}"
echo "Log Dir: ${log_dir}"

python run.py "${py_script}" -w "${work_dir}" \
>> "${log_dir}/eval-${task}-${timestamp}.out" \
2>> "${log_dir}/eval-${task}-${timestamp}.err" &
