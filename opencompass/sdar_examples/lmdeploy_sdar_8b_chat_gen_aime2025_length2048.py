from mmengine.config import read_base
with read_base():
    from ..opencompass.configs.datasets.aime2025.aime2025_gen_repeat8_5e9f4f import \
        aime2025_datasets
    from ..opencompass.configs.models.dllm.lmdeploy_sdar_8b_chat import \
        models as lmdeploy_sdar_8b_chat
    from ..opencompass.configs.summarizers.groups.OlympiadBench import \
        OlympiadBenchMath_summary_groups
    
datasets = aime2025_datasets
models = lmdeploy_sdar_8b_chat
summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], []
)
summary_groups.extend([
    {
        'name': 'AIME2025-Aveage8',
        'subsets':[[f'aime2025-run{idx}', 'accuracy'] for idx in range(8)]
    }
])
summarizer = dict(
    dataset_abbrs=[
        ['AIME2025-Aveage8', 'naive_average'],
    ],
    summary_groups=summary_groups,
)
eval_cfg = {
    'engine_config': {
        'session_len': 4096, 
        'max_batch_size': 16, 
        'tp': 1,
        'dtype': "float16",
        'max_prefill_token_num': 2048,
        'cache_max_entry_count': 0.8,
        'dllm_block_length': 4,
        'dllm_denoising_steps': 4,
        'dllm_unmasking_strategy': "low_confidence_dynamic",
        'dllm_confidence_threshold': 0.9,
    },
    'gen_config': {
        'top_k': 0, 
        'temperature': 1.0, 
        'top_p': 0.95, 
        'do_sample': False, 
        'max_new_tokens': 2048,
    },
    'max_seq_len': 4096,
    'max_out_len': 2048,
    'batch_size': 16,
}

for model in models:
    model.update(eval_cfg)
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=8,    
        num_split=None,   
        min_task_size=16, 
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        retry=5
    ),
)

