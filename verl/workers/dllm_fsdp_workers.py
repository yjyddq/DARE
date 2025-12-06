# Copyright 2025 Shanghai AI Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings
from typing import Union

import psutil
import torch
import torch.distributed
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from torch.distributed.device_mesh import init_device_mesh

import verl.utils.torch_functional as verl_F
from verl.utils.py_functional import convert_to_regular_types
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    layered_summon_lora_params,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.workers.fsdp_workers import get_sharding_strategy, ActorRolloutRefWorker

from peft import LoraConfig, TaskType, get_peft_model
from codetiming import Timer

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from peft import PeftModel
from safetensors.torch import save_file
from dataclasses import asdict
import json
from tensordict import TensorDict

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class DLLMActorRolloutRefWorker(ActorRolloutRefWorker):
    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
        attn_implementation="eager",  # LNY: Force using the most basic PyTorch Attention implementation
    ):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(
            local_path, 
            trust_remote_code=trust_remote_code, 
            attn_implementation=attn_implementation,
        )  # LNY: LLaDA does not support flash-attn?
                
        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            else:
                actor_module_class = AutoModelForCausalLM

            # # LNY: ref model enable quantization
            # bnb_config = None
            # if role == "ref":
            #     from transformers import BitsAndBytesConfig
            #     bnb_config = BitsAndBytesConfig(
            #         load_in_4bit=True,
            #         bnb_4bit_use_double_quant=True,
            #         bnb_4bit_quant_type="nf4",
            #         bnb_4bit_compute_dtype=torch.bfloat16,
            #     )
                
            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
                # quantization_config=bnb_config,  # LNY: ref model enable quantization
            )

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if self._is_lora:
                print("Applying LoRA to actor module")
                actor_module.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    'task_type': TaskType.CAUSAL_LM,
                    'r': self.config.model.lora_rank,
                    'lora_alpha': self.config.model.lora_alpha,
                    'target_modules': convert_to_regular_types(self.config.model.target_modules),
                    'bias': "none",
                    'lora_dropout': self.config.model.lora_dropout,  # LNY: add dropout
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
                actor_module.print_trainable_parameters()
                
                # LNY: After injecting LoRA, ensure all parameters use torch_dtype
                for param in actor_module.parameters():
                    if param.dtype != torch_dtype:
                        param.data = param.data.to(torch_dtype)
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get("wrap_policy", None), is_lora=self.config.model.get('lora_rank', 0) > 0)

        if self._is_rollout and self.config.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=self._is_lora,  # LNY: When using LoRA, use_orig_params=True must be set, because the LoRA adapter needs to access the original parameters
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            warmup_style = optim_config.get("warmup_style", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if warmup_style == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps)
            elif warmup_style == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps, min_lr_ratio=min_lr_ratio, num_cycles=num_cycles)
            else:
                raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.rollout.name
        use_cache = self.config.rollout.use_cache
        if rollout_name == "hf":
            if not use_cache:
                if self.config.algorithm.name == "cj-grpo":
                    from verl.workers.rollout.dllm_rollout_traj import DLLMRollout
                elif self.config.algorithm.name == "mdpo":
                    from verl.workers.rollout.mdpo_rollout import DLLMRollout
                else:
                    from verl.workers.rollout.dllm_rollout import DLLMRollout
                from verl.workers.sharding_manager.base import BaseShardingManager

                rollout = DLLMRollout(module=self.actor_module_fsdp, config=self.config.rollout, tokenizer=self.tokenizer)
                rollout_sharding_manager = BaseShardingManager()
                # TODO: a sharding manager that do nothing?
            else:
                if self.config.algorithm.name == "cj-grpo":
                    from verl.workers.rollout.fast_dllm_rollout_traj import FASTDLLMRollout
                elif self.config.algorithm.name == "mdpo":
                    from verl.workers.rollout.fast_mdpo_rollout import FASTDLLMRollout
                else:
                    from verl.workers.rollout.fast_dllm_rollout import FASTDLLMRollout
                from verl.workers.sharding_manager.base import BaseShardingManager

                rollout = FASTDLLMRollout(module=self.actor_module_fsdp, config=self.config.rollout, tokenizer=self.tokenizer)
                rollout_sharding_manager = BaseShardingManager()
                # TODO: a sharding manager that do nothing?

        elif rollout_name == "vllm":    # !!! We have not yet adapted dLLM to vllm !!!
            from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get('use_shm', False))
            lora_kwargs = {'lora_kwargs': {"enable_lora":True, "max_loras":1, "max_lora_rank":self._lora_rank}} if self._is_lora else {}
            # lora_kwargs = {}
            if vllm_mode == "customized":
                rollout = vLLMRollout(
                    actor_module=self.actor_module_fsdp,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    trust_remote_code=trust_remote_code,
                    **lora_kwargs)
            elif vllm_mode == "spmd":
                from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

                vllm_rollout_cls = vLLMRollout if self.config.rollout.mode == "sync" else vLLMAsyncRollout
                rollout = vllm_rollout_cls(
                    model_path=local_path,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    device_mesh=rollout_device_mesh,
                    trust_remote_code=trust_remote_code,
                    **lora_kwargs)
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")

            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
            full_params = torch.distributed.get_world_size() == 1
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get('layered_summon', False),
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif rollout_name in ["sglang", "sglang_async"]:    # !!! We have not yet adapted dLLM to sglang !!!
            if rollout_name == "sglang_async":
                warnings.warn(
                    "'sglang_async' has been deprecated and merged into 'sglang'. "
                    "Please use 'sglang' going forward.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            from verl.workers.rollout.sglang_rollout import SGLangRollout
            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
            # SGLang's model_runner would check CUDA device capability. However, due to verl's setting,
            # the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
            # we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76

            from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

            local_path = copy_to_local(self.config.model.path)
            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
            )
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        else:
            raise NotImplementedError(f"Rollout name: {self.config.rollout.name} is not supported")

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.config.algorithm.name == 'spg':
            from verl.workers.actor.dllm_dp_actor_spg import DLLMDataParallelPPOActor
        else:
            from verl.workers.actor.dllm_dp_actor import DLLMDataParallelPPOActor
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get('use_shm', False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
                attn_implementation=self.config.model.get("attn_implementation", "eager"),  # LNY: Force using the most basic PyTorch Attention implementation
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.use_fused_kernels = use_fused_kernels
            self.actor = DLLMDataParallelPPOActor(config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DLLMDataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_contents=self.config.actor.checkpoint.contents,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(get_torch_device().current_device())

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            
            log_gpu_memory_usage("After rollout generation", logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def forward_process(self, prompts: DataProto):
        """
        forward process: add noise to the batch. Only mask the part where attention_mask == 1, padding part and prompt part are not masked
        
        Args:
            prompts: DataProto containing batch prompts
            
        Returns:
            DataProto with perturbed sequences, computed log probabilities and related tensors
        """
        batch = prompts.batch
        # Extract data from batch
        idx_repeat = batch["prompts"]  # (batch_size * n_rollout, prompt_len)
        responses = batch["responses"]  # (batch_size * n_rollout, response_len)
        input_ids = batch["input_ids"]  # (batch_size * n_rollout, seq_len)
        attention_mask = batch["attention_mask"]  # (batch_size * n_rollout, seq_len)
        position_ids = batch["position_ids"]  # Complete position_ids, prompt is left-padded, response is right-padded
        response_length = self.config.rollout.get("response_length")
        total_batch_size  = input_ids.shape[0]

        # Get parameters from rollout config
        n_l = self.config.actor.get("n_l", 1)
        mc_num = self.config.actor.get("mc_num", 1)
        MASK_TOKEN_ID = self.actor_module_fsdp.config.mask_token_id

        # select _forward_process according to algorithm
        if self.config.algorithm.name in ["d1", "bgpo", "coupled-grpo"]:
            if self.config.algorithm.name == "d1":
                assert n_l == mc_num == 1, "d1 method requires n_l == mc_num == 1"
                from verl.trainer.ppo.dllm_core_algos import _forward_process_d1 as _forward_process
            elif self.config.algorithm.name == "coupled-grpo":
                assert n_l == mc_num == 1, "coupled-grpo method requires n_l == mc_num == 1"
                from verl.trainer.ppo.dllm_core_algos import _forward_process_coupled_grpo as _forward_process
            elif self.config.algorithm.name == "bgpo":
                from verl.trainer.ppo.dllm_core_algos import _forward_process_bgpo as _forward_process

            batch_size, seq_len = input_ids.shape
            prompt_len = seq_len - response_length  # int
            device = input_ids.device
            n_y_l = mc_num // n_l  # mc_num: Monte Carlo sampling times

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):            
                # Generate perturbed_seq, mask_indices, p_mask for each sample in the batch
                all_perturbed_seqs = []
                all_mask_indices = []
                all_p_mask = []

                for i in range(batch_size):
                    single_input_id = input_ids[i:i+1].repeat((n_l, 1)).to(device)  # (n_l, seq_len) 

                    mc_perturbed_seq_list = []
                    mc_mask_indices_list = []
                    mc_p_mask_list = []
                    
                    for j in range(n_y_l):
                        perturbed_seq, mask_indices, p_mask = _forward_process(batch=single_input_id, attention_mask=attention_mask[i], prompt_len=prompt_len, MASK_TOKEN_ID=MASK_TOKEN_ID)  # (n_l, seq_len)
                        assert (mask_indices == (perturbed_seq == MASK_TOKEN_ID)).all()
                        
                        mc_perturbed_seq_list.append(perturbed_seq)
                        mc_mask_indices_list.append(mask_indices)
                        mc_p_mask_list.append(p_mask)
                    
                    all_perturbed_seqs.append(torch.cat(mc_perturbed_seq_list, dim=0))  # (mc_num, seq_len)
                    all_mask_indices.append(torch.cat(mc_mask_indices_list, dim=0))  # (mc_num, seq_len)
                    all_p_mask.append(torch.cat(mc_p_mask_list, dim=0))  # (mc_num, seq_len)
                
                perturbed_seq = torch.stack(all_perturbed_seqs, dim=0)  # (batch_size, mc_num, seq_len)
                mask_indices = torch.stack(all_mask_indices, dim=0)  # (batch_size, mc_num, seq_len)
                p_mask = torch.stack(all_p_mask, dim=0)  # (batch_size, mc_num, seq_len)

        elif self.config.algorithm.name == "spg":
            from verl.trainer.ppo.dllm_core_algos import _forward_process_spg as _forward_process
            
            block_length = self.config.rollout["block_length"]
            
            batch_size, seq_len = input_ids.shape
            prompt_len = seq_len - response_length  # int
            device = input_ids.device
            n_y_l = mc_num // n_l  # mc_num: Monte Carlo sampling times

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):            
                # Generate perturbed_seq, mask_indices, p_mask for each sample in the batch
                all_perturbed_seqs = []
                all_mask_indices = []
                all_p_mask = []

                for i in range(batch_size):
                    single_input_id = input_ids[i:i+1].repeat((1, 1)).to(device)  # (1, seq_len) 

                    mc_perturbed_seq_list = []
                    mc_mask_indices_list = []
                    mc_p_mask_list = []
                    
                    for j in range(n_y_l):
                        perturbed_seq, mask_indices, p_mask = _forward_process(batch=single_input_id, attention_mask=attention_mask[i],  prompt_len=prompt_len, block_length=block_length, num_t=n_l, MASK_TOKEN_ID=MASK_TOKEN_ID)  # (n_l, seq_len)
                        assert (mask_indices == (perturbed_seq == MASK_TOKEN_ID)).all()
                        
                        mc_perturbed_seq_list.append(perturbed_seq)
                        mc_mask_indices_list.append(mask_indices)
                        mc_p_mask_list.append(p_mask)
                    
                    all_perturbed_seqs.append(torch.cat(mc_perturbed_seq_list, dim=0))  # (mc_num, seq_len)
                    all_mask_indices.append(torch.cat(mc_mask_indices_list, dim=0))  # (mc_num, seq_len)
                    all_p_mask.append(torch.cat(mc_p_mask_list, dim=0))  # (mc_num, seq_len)
                
                perturbed_seq = torch.stack(all_perturbed_seqs, dim=0)  # (batch_size, mc_num, seq_len)
                mask_indices = torch.stack(all_mask_indices, dim=0)  # (batch_size, mc_num, seq_len)
                p_mask = torch.stack(all_p_mask, dim=0)  # (batch_size, mc_num, seq_len)

        else:
            NotImplementedError(f"Unsupported algorithm: {self.config.algorithm.name} for forward process in DLLMActorRolloutRefWorker")
        
        

        
        batch = TensorDict(
            {
                "prompts": idx_repeat,
                "responses": responses,
                "input_ids": input_ids,  # Complete prompt + response
                "attention_mask": attention_mask,
                "position_ids": position_ids,  # Complete position_ids, prompt is left-padded, response is right-padded
                "perturbed_seq": perturbed_seq,  # (batch_size * n_rollout, mc_num, seq_len)
                "mask_indices": mask_indices,
                "p_mask": p_mask,
            },
            batch_size=total_batch_size,
        )
        
        return DataProto(batch=batch)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext
        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        data = data.to(get_torch_device().current_device())
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            with adapter_ctx:
                output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output, "entropys": entropys},
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output