from mmengine.config import read_base
with read_base():
    from ..opencompass.configs.datasets.mbpp.mbpp_gen import \
        mbpp_datasets
    from ..opencompass.configs.models.dllm.dream_v0_instruct_7b import \
        models as dream_v0_instruct_7b
datasets = mbpp_datasets
models = dream_v0_instruct_7b
eval_cfg = {
    'gen_length': 512, 
    'gen_steps': 512, 
    'batch_size': 1, 
    'batch_size_': 1,
    'model_kwargs': {
        'attn_implementation': 'flash_attention_2',  #'sdpa'
        'torch_dtype': 'bfloat16',
        'device_map': 'auto',
        'trust_remote_code': True,
    },
    'temperature': 0.2,
    'top_p': 0.95,
    'alg': 'entropy'
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
