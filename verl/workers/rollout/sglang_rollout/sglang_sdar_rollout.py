# Copyright 2025 Shanghai AI Lab Ltd. and/or its affiliates
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
SGLang Rollout Worker for SDAR (Block Diffusion) models.

This module provides SGLang-based rollout functionality for SDAR models,
which use block-level diffusion rather than token-level autoregressive generation.
It integrates with SGLang's built-in dLLM (diffusion LLM) support.
"""

import logging
import os
from typing import TYPE_CHECKING

from omegaconf import DictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SGLangSDARRollout(SGLangRollout):
    """SGLang rollout worker for SDAR (block diffusion) models.

    This class extends SGLangRollout to support SDAR's block diffusion forward pass.
    It leverages SGLang's built-in dLLM framework for efficient inference.

    SDAR-specific parameters:
        - mask_token_id: Token ID used for masking (default: 151669 for 8B model)
        - block_length: Block size for diffusion (default: 4)
        - dllm_algorithm: dLLM algorithm to use (e.g., "LowConfidence", "JointThreshold")

    Example config:
        ```yaml
        actor_rollout_ref:
          rollout:
            name: "sglang"
            model_name: "sdar"
            mask_token_id: 151669
            block_length: 4
            dllm_algorithm: "LowConfidence"
            mem_fraction_static: 0.6
            max_running_requests: 1
            attention_backend: "flashinfer"
        ```
    """

    def __init__(
        self,
        actor_module: str,
        config: DictConfig,
        tokenizer: "PreTrainedTokenizer",
        model_hf_config,
        port=None,
        trust_remote_code: bool = False,
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ):
        """Initialize SGLangSDARRollout.

        Args:
            actor_module: Path to the SDAR model (HuggingFace format)
            config: Configuration DictConfig containing:
                - mask_token_id: SDAR mask token ID (default: 151669)
                - block_length: Block size for diffusion (default: 4)
                - dllm_algorithm: dLLM algorithm (default: "LowConfidence")
                - mem_fraction_static: GPU memory fraction (default: 0.6)
                - max_running_requests: Max concurrent requests (default: 1)
                - attention_backend: Attention backend (default: "flashinfer")
            tokenizer: Tokenizer compatible with the model
            model_hf_config: HuggingFace model configuration
            port: Port for multi-node setup
            trust_remote_code: Whether to trust remote code
            device_mesh: Device mesh for distributed setup
            **kwargs: Additional arguments (e.g., train_tp for Megatron)
        """
        # Store SDAR-specific parameters before calling parent init
        self._mask_token_id = config.get("mask_token_id", 151669)
        self._block_length = config.get("block_length", 4)
        self._dllm_algorithm = config.get("dllm_algorithm", "LowConfidence")
        self._dllm_algorithm_config = config.get("dllm_algorithm_config", None)

        logger.info(
            f"Initializing SGLangSDARRollout with: "
            f"mask_token_id={self._mask_token_id}, "
            f"block_length={self._block_length}, "
            f"dllm_algorithm={self._dllm_algorithm}, "
            f"dllm_algorithm_config={self._dllm_algorithm_config}"
        )

        # Call parent init
        super().__init__(
            actor_module=actor_module,
            config=config,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            port=port,
            trust_remote_code=trust_remote_code,
            device_mesh=device_mesh,
            **kwargs,
        )

    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        """Initialize SGLang engine with SDAR/dLLM support.

        This method overrides the parent to add SDAR-specific parameters:
        - dllm_algorithm: The dLLM algorithm to use
        - These parameters are passed to SGLang's Engine initialization

        SGLang will automatically detect the SDAR model architecture and apply
        the appropriate block diffusion forward pass.
        """
        from sglang.srt.entrypoints.engine import Engine
        from .utils import broadcast_pyobj, get_ip, get_open_port

        # Initialize distributed environment
        nnodes = -(-self._tp_size // len(self.visible_devices_set))
        if nnodes > 1:
            ip = get_ip()
            port = get_open_port() if port is None else port
            [ip, port] = broadcast_pyobj(
                [ip, port],
                rank=self._rank,
                dist_group=self._device_mesh_cpu.get_group(self._tp_mesh_name),
                src=self._device_mesh_cpu[self._tp_mesh_name].mesh[0].item(),
                force_cpu_device=False,
            )
            from verl.workers.rollout.sglang_rollout.utils import is_ipv6
            dist_init_addr = f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"
        else:
            dist_init_addr = None

        load_format = "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        if first_rank_in_node:
            rank = self._rank
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

            # Build engine arguments with SDAR/dLLM specific parameters
            engine_kwargs = dict(
                model_path=actor_module,
                dtype=self.config.dtype,
                mem_fraction_static=self.config.get("mem_fraction_static", 0.6),
                enable_memory_saver=True,
                base_gpu_id=0,
                gpu_id_step=1,
                tp_size=self._tp_size,
                node_rank=node_rank,
                load_format=load_format,
                dist_init_addr=dist_init_addr,
                nnodes=nnodes,
                trust_remote_code=trust_remote_code,
                port=30000 + rank,
                max_running_requests=self.config.get("max_running_requests", 1),
            )

            # Add SDAR/dLLM specific parameters
            engine_kwargs["dllm_algorithm"] = self._dllm_algorithm
            if self._dllm_algorithm_config is not None:
                engine_kwargs["dllm_algorithm_config"] = self._dllm_algorithm_config

            # Optionally add attention backend if specified
            if self.config.get("attention_backend"):
                engine_kwargs["attention_backend"] = self.config.get("attention_backend")

            logger.info(f"Initializing SGLang Engine for SDAR with dllm_algorithm={self._dllm_algorithm}")

            self._engine = Engine(**engine_kwargs)
        else:
            self._engine = None

        self.sharding_manager = None
        if self._tp_rank == 0:
            self._engine.release_memory_occupation()
        self.is_sleep = True

    @property
    def mask_token_id(self) -> int:
        """Return the SDAR mask token ID."""
        return self._mask_token_id

    @property
    def block_length(self) -> int:
        """Return the SDAR block length."""
        return self._block_length

    @property
    def dllm_algorithm(self) -> str:
        """Return the dLLM algorithm being used."""
        return self._dllm_algorithm

    @property
    def dllm_algorithm_config(self):
        """Return the dLLM algorithm configuration passed to SGLang."""
        return self._dllm_algorithm_config
