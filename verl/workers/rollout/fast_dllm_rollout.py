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
In single GPU rollout, the sequences are generated directly by sampling from the model.
The output will contain
1. output_ids
2. attention_masks (left padding)
3. eos_masks
4. log_probs
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from tensordict import TensorDict
from torch import nn
import re
import time

from verl import DataProto

from verl.workers.rollout.base import BaseRollout
from .rollout_utils import execute_fastdllm_generation

__all__ = ["DLLMRollout"]


class FASTDLLMRollout(BaseRollout):
    def __init__(self, module: nn.Module, config, tokenizer):
        """A naive rollout for Diffusion LLM. Requires HuggingFace-style module."""
        super().__init__()
        self.config = config
        self.module = module
        self.tokenizer = tokenizer
        self.MASK_TOKEN_ID = self.module.config.mask_token_id
        self.PAD_TOKEN_ID = self.module.config.pad_token_id
        self.EOS_TOKEN_ID = self.module.config.eos_token_id

        # diffusion related parameters
        self.response_length = config["response_length"]  # Response length
        self.num_diffusion_steps = config["num_diffusion_steps"]  # Number of diffusion steps
        self.block_length = config["block_length"]  # Block length
        self.mc_num = config["mc_num"]  # Number of Monte Carlo samples
        self.n_l = config["n_l"]  # Number of random masks
        self.cfg_scale = config["cfg_scale"]  # Whether to use CFG
        # cache in fastdllm
        self.use_cache = config["use_cache"]  # Number of random masks
        self.dual_cache = config["dual_cache"]  # Whether to use CFG

        # rollout related parameters
        self.n_rollout = config["n"]  # How many responses to generate for each prompt
        self.temperature = config["temperature"]  # Temperature during training
        self.val_kwargs = config["val_kwargs"]  # Validation generation parameters

    # from .auto_line_tracker import auto_track_lines
    # @auto_track_lines(interval=10.0)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences using diffusion LLM (LLaDA style)
        1. For each prompt, generate n_rollout responses y
        2. For each y, perform multiple random masks and regenerate, then use ELBO to approximate the loglikelihood of y
        """
        # Start timer
        start_time = time.time()
        
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info.get("eos_token_id", self.EOS_TOKEN_ID)
        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        self.module.eval()

        is_validate = prompts.meta_info.get("validate", False)  # Only ray_trainer._validate sets this parameter
        
        # Select generation parameters based on mode
        n_rollout = 1 if is_validate else self.n_rollout  # In validation stage, the input has been repeated val_kwargs.n times in ray_trainer._validate
        gen_kwargs = {
            "steps": self.val_kwargs["num_diffusion_steps"] if is_validate else self.num_diffusion_steps,
            "gen_length": self.response_length,
            "block_length": self.block_length,
            "temperature": self.val_kwargs.get("temperature", self.temperature) if is_validate else self.temperature,
            # "cfg_scale": self.cfg_scale,
            "dual_cache": self.dual_cache,
            "remasking": "low_confidence",
            "mask_id": self.MASK_TOKEN_ID,
            "mode": "train" if not is_validate else "eval",
        }
        print(f"gen_kwargs: {gen_kwargs}")

        idx_repeat = idx.repeat_interleave(n_rollout, dim=0)
        attention_mask_repeat = attention_mask.repeat_interleave(n_rollout, dim=0)

        MAX_MODEL_LENGTH = self.config.max_num_batched_tokens  # Maximum length of packed sequences
        total_batch_size = batch_size * n_rollout

        responses, full_input_ids, attention_mask, answers = execute_fastdllm_generation(
            idx_repeat=idx_repeat,
            module=self.module,
            attention_mask_repeat=attention_mask_repeat,
            response_length=self.response_length,
            tokenizer=self.tokenizer,
            gen_kwargs=gen_kwargs
        )
    
        if not is_validate:
            for j in range(total_batch_size):
                print(f"==================[RANK{dist.get_rank()}] rollout question ID: {j}=================\nGenerated answer: {answers[j]}\n==========================================")
        else:
            for j in range(total_batch_size):
                print(f"==================[RANK{dist.get_rank()}] validation question ID: {j}=================\nGenerated answer: {answers[j]}\n==========================================")

        input_ids = torch.cat([idx_repeat, responses], dim=1)
        batch = TensorDict(
            {
                "prompts": idx_repeat,
                "responses": responses,
                "input_ids": input_ids,  # Complete prompt + response
                "attention_mask": attention_mask,
                "position_ids": torch.cat([position_ids.repeat_interleave(n_rollout, dim=0), 
                                         position_ids[:, -1:].repeat_interleave(n_rollout, dim=0) + 
                                         torch.arange(1, self.response_length+1, device=position_ids.device)], dim=1),  # Complete position_ids, prompt is left-padded, response is right-padded
            },
            batch_size=total_batch_size,
        )
        
        self.module.train()
        
        total_time = time.time() - start_time
        print(f"[RANK{dist.get_rank()}] generate_sequences total time: {total_time:.2f}s")

        return DataProto(batch=batch)
