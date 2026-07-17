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
LLaDA2.x EBPO actor.

Like the LLaDA2 BGPO path, EBPO uses the official vectorized block-diffusion
forward over ``[noisy_x, clean_x]``. Each forward returns the summed ELBO
contributions from every response-overlapping block.
"""

import logging
import os

import torch

from verl.workers.actor.block_diffusion_utils import scatter_compact_values_to_response
from verl.workers.actor.ebpo_actor_mixin import EBPOActorMixin
from verl.workers.actor.llada2_dp_actor_bgpo import DLLMDataParallelPPOActor as BGPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DLLMDataParallelPPOActor(EBPOActorMixin, BGPOActor):
    def _forward_micro_batch(self, micro_batch, temperature, n_l, mc_num, calculate_entropy=False, call_fn_name=""):
        batch_size = micro_batch["input_ids"].size(0)
        response_length = micro_batch["responses"].size(-1)
        device = micro_batch["input_ids"].device

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            perturbed_seq = micro_batch["perturbed_seq"]
            mask_indices = micro_batch["mask_indices"]
            p_mask = micro_batch["p_mask"]
            mc_num = perturbed_seq.size(1)
            loss_per_token = torch.zeros((batch_size, mc_num, response_length), device=device)
            for i in range(mc_num):
                artifacts = self._build_block_diffusion_artifacts(
                    micro_batch=micro_batch,
                    noisy_input_ids=perturbed_seq[:, i, :],
                    target_mask=mask_indices[:, i, :],
                    p_mask=p_mask[:, i, :],
                )
                token_losses = self._compute_block_diffusion_token_losses(artifacts)
                loss_per_token[:, i, :] = scatter_compact_values_to_response(
                    -token_losses,
                    artifacts,
                    response_length=response_length,
                )

            log_likelihood = loss_per_token.mean(dim=1).sum(dim=-1)
            log_prob = log_likelihood.unsqueeze(-1).expand(-1, response_length) # (batch_size, response_length)

        entropy = None
        if calculate_entropy:
            entropy = -log_prob.exp() * log_prob

        return entropy, log_prob, loss_per_token
