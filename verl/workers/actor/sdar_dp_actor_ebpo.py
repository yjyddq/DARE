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
SDAR EBPO actor.

EBPO uses the same external forward-process artifacts as SDAR BGPO. The SDAR
model already vectorizes all block-conditional terms, so this actor restores
the unnormalized all-block ELBO from the model's response-normalized loss and
feeds it to the shared composite-ratio objective.
"""

import logging
import os
from typing import Tuple

import torch

from verl.workers.actor.ebpo_actor_mixin import EBPOActorMixin
from verl.workers.actor.sdar_dp_actor_bgpo import DLLMDataParallelPPOActor as BGPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _restore_sdar_sequence_elbo(normalized_nll: torch.Tensor, valid_response_count: int) -> torch.Tensor:
    """Undo SDAR's ``sum(token_nll) / answer_len`` normalization."""
    if valid_response_count <= 0:
        raise ValueError(f"valid_response_count must be positive, got {valid_response_count}")
    return -normalized_nll * valid_response_count


class DLLMDataParallelPPOActor(EBPOActorMixin, BGPOActor):
    def __init__(self, config, actor_module, actor_optimizer=None):
        super().__init__(config, actor_module, actor_optimizer)
        model_block_length = int(actor_module.config.block_size)
        self.block_length = int(config.get("block_length", model_block_length))
        if self.block_length != model_block_length:
            raise ValueError(
                "SDAR EBPO actor.block_length must match model.config.block_size, "
                f"got {self.block_length} and {model_block_length}"
            )

    def _forward_micro_batch(self, micro_batch, temperature, n_l, mc_num, calculate_entropy=False, call_fn_name="") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_length = micro_batch["input_ids"].size(0), micro_batch["input_ids"].size(-1)
        response_length = micro_batch["responses"].size(-1)
        prompt_length = seq_length - response_length
        device = micro_batch["input_ids"].device

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            position_ids = micro_batch["position_ids"]
            seq = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            perturbed_seq = micro_batch["perturbed_seq"]
            mask_indices = micro_batch["mask_indices"]
            p_mask = micro_batch["p_mask"]
            mc_num = perturbed_seq.shape[1]
            loss_per_token = torch.zeros((batch_size, mc_num, response_length), device=device)
            for b in range(batch_size):
                response_mask = attention_mask[b, -response_length:].bool()
                valid_response_count = int(response_mask.sum().item())
                if valid_response_count == 0:
                    continue
                for i in range(mc_num):
                    response_target_mask = mask_indices[b, i, -response_length:].bool() & response_mask
                    if not response_target_mask.any():
                        continue

                    loss_b_i = self._get_logits(
                        model=self.actor_module,
                        seq=seq[b:b+1, :],
                        attention_mask=attention_mask[b:b+1, :],
                        position_ids=position_ids[b:b+1, :],
                        prompt_len=prompt_length,
                        perturbed_seq=perturbed_seq[b:b+1, i, :],
                        mask_indices=mask_indices[b:b+1, i, :],
                        p_mask=p_mask[b:b+1, i, :],
                        cfg_scale=0.0,
                        MASK_TOKEN_ID=self.MASK_TOKEN_ID,
                    )
                    # SDAR divides the summed diffusion NLL by the number of
                    # non-ignored labels. Undo that exact denominator; using the
                    # padded response_length is wrong whenever response padding
                    # is present.
                    sequence_elbo = _restore_sdar_sequence_elbo(loss_b_i, valid_response_count)
                    loss_per_token[b, i, response_mask] = (
                        sequence_elbo.to(loss_per_token.dtype) / valid_response_count
                    )

            log_likelihood = loss_per_token.mean(dim=1).sum(dim=-1)
            log_prob = log_likelihood.unsqueeze(-1).expand(-1, response_length)

        entropy = None
        if calculate_entropy:
            entropy = -log_prob.exp() * log_prob

        return entropy, log_prob, loss_per_token
