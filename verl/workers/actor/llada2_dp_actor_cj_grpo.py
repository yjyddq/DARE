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
"""LLaDA2 CJ-GRPO actor."""

from verl.workers.actor.cj_grpo_sglang_mixin import CJGRPOActorMixin
from verl.workers.actor.llada2_dp_actor_bgpo import (
    DLLMDataParallelPPOActor as LLaDA2BGPOActor,
)

__all__ = ["DLLMDataParallelPPOActor"]


class DLLMDataParallelPPOActor(CJGRPOActorMixin, LLaDA2BGPOActor):
    pass
