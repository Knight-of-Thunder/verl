# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Focused FSDP2 model-transfer smoke test.

Run with:
    torchrun --standalone --nproc-per-node=6 tests/special_distributed/run_fsdp2_resize_transfer.py

Set ``OLD_WORLD_SIZE=6 NEW_WORLD_SIZE=4`` to exercise actor scale-down.
"""

import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard

from verl.utils.fsdp_resize import (
    restore_actor_rng_state,
    restore_fsdp2_model_state,
    restore_fsdp2_optimizer_state,
    restore_fsdp2_scheduler_state,
    stage_actor_rng_state,
    stage_fsdp2_model_state,
    stage_fsdp2_optimizer_state,
    stage_fsdp2_scheduler_state,
)
from verl.utils.torch_functional import get_constant_schedule_with_warmup


def _build_model() -> nn.Module:
    torch.manual_seed(1234)
    model = nn.Sequential(nn.Linear(4, 12), nn.GELU(), nn.Linear(12, 3, bias=False))
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.copy_(torch.arange(parameter.numel()).reshape(parameter.shape) + index * 1000)
    return model.cuda()


def _optimizer_step(model: nn.Module, optimizer: torch.optim.Optimizer, step_index: int) -> None:
    optimizer.zero_grad(set_to_none=True)
    for parameter_index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, 0.125 * (step_index + parameter_index + 1))
    optimizer.step()


def _build_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=4)


def _seed_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _draw_rng() -> tuple[float, float, torch.Tensor, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(4), torch.rand(4, device="cuda").cpu()


def _assert_model_matches(actual_model: nn.Module, expected_model: nn.Module) -> None:
    expected_state = expected_model.state_dict()
    for name, value in actual_model.state_dict().items():
        actual = value.full_tensor() if hasattr(value, "full_tensor") else value
        torch.testing.assert_close(actual, expected_state[name])


def _assert_optimizer_matches(
    actual_model: nn.Module,
    actual_optimizer: torch.optim.Optimizer,
    expected_model: nn.Module,
    expected_optimizer: torch.optim.Optimizer,
) -> None:
    expected_parameters = dict(expected_model.named_parameters())
    for name, actual_parameter in actual_model.named_parameters():
        actual_state = actual_optimizer.state[actual_parameter]
        expected_state = expected_optimizer.state[expected_parameters[name]]
        assert actual_state.keys() == expected_state.keys()
        for state_name, actual_value in actual_state.items():
            expected_value = expected_state[state_name]
            if hasattr(actual_value, "full_tensor"):
                actual_value = actual_value.full_tensor()
            torch.testing.assert_close(actual_value, expected_value)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    old_world_size = int(os.environ.get("OLD_WORLD_SIZE", "4"))
    new_world_size = int(os.environ.get("NEW_WORLD_SIZE", "6"))
    old_ranks = list(range(old_world_size))
    new_ranks = list(range(new_world_size))
    assert dist.get_world_size() == max(old_world_size, new_world_size)

    old_group = dist.new_group(ranks=old_ranks)
    old_model = None
    old_optimizer = None
    old_scheduler = None
    if rank in old_ranks:
        old_mesh = DeviceMesh.from_group(
            old_group,
            device_type="cuda",
            mesh=torch.arange(len(old_ranks)),
            mesh_dim_names=["fsdp"],
        )
        old_model = _build_model()
        fully_shard(old_model, mesh=old_mesh)
        old_optimizer = torch.optim.AdamW(old_model.parameters(), lr=2e-3, weight_decay=0.03)
        old_scheduler = _build_scheduler(old_optimizer)
        _optimizer_step(old_model, old_optimizer, step_index=0)
        old_scheduler.step()

    staged_state, _ = stage_fsdp2_model_state(
        old_model,
        old_ranks=old_ranks,
        new_ranks=new_ranks,
        transfer_group=dist.group.WORLD,
        device=torch.device("cuda", local_rank),
    )
    assert (staged_state is not None) == (rank in new_ranks)
    staged_optimizer_state, _ = stage_fsdp2_optimizer_state(
        old_model,
        old_optimizer,
        old_ranks=old_ranks,
        new_ranks=new_ranks,
        transfer_group=dist.group.WORLD,
        device=torch.device("cuda", local_rank),
    )
    assert (staged_optimizer_state is not None) == (rank in new_ranks)
    staged_scheduler_state, _ = stage_fsdp2_scheduler_state(
        old_scheduler,
        old_ranks=old_ranks,
        new_ranks=new_ranks,
        transfer_group=dist.group.WORLD,
        device=torch.device("cuda", local_rank),
    )
    assert (staged_scheduler_state is not None) == (rank in new_ranks)
    _seed_rng(7000 + rank)
    staged_rng_state = stage_actor_rng_state(
        old_ranks=old_ranks,
        new_ranks=new_ranks,
        global_rank=rank,
        base_seed=42,
        global_step=5,
    )
    assert (staged_rng_state is not None) == (rank in new_ranks)
    expected_rng_draw = None
    if staged_rng_state is not None:
        restore_actor_rng_state(staged_rng_state)
        expected_rng_draw = _draw_rng()

    del old_model
    del old_optimizer
    del old_scheduler
    if rank in old_ranks:
        dist.destroy_process_group(old_group)
    torch.cuda.empty_cache()
    dist.barrier()

    new_group = dist.new_group(ranks=new_ranks)
    if rank in new_ranks:
        new_mesh = DeviceMesh.from_group(
            new_group,
            device_type="cuda",
            mesh=torch.arange(len(new_ranks)),
            mesh_dim_names=["fsdp"],
        )
        new_model = _build_model()
        with torch.no_grad():
            for parameter in new_model.parameters():
                parameter.zero_()
        fully_shard(new_model, mesh=new_mesh)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=9e-4, weight_decay=0.0)
        new_scheduler = _build_scheduler(new_optimizer)
        assert staged_state is not None
        assert staged_optimizer_state is not None
        assert staged_scheduler_state is not None
        assert staged_rng_state is not None
        assert expected_rng_draw is not None
        restore_fsdp2_model_state(new_model, staged_state)
        restore_fsdp2_optimizer_state(new_model, new_optimizer, staged_optimizer_state)
        restore_fsdp2_scheduler_state(new_scheduler, staged_scheduler_state)

        expected_model = _build_model()
        expected_optimizer = torch.optim.AdamW(expected_model.parameters(), lr=2e-3, weight_decay=0.03)
        expected_scheduler = _build_scheduler(expected_optimizer)
        _optimizer_step(expected_model, expected_optimizer, step_index=0)
        expected_scheduler.step()
        _assert_model_matches(new_model, expected_model)
        _assert_optimizer_matches(new_model, new_optimizer, expected_model, expected_optimizer)
        assert new_scheduler.state_dict() == expected_scheduler.state_dict()
        assert new_scheduler.get_last_lr() == expected_scheduler.get_last_lr()

        restore_actor_rng_state(staged_rng_state)
        actual_rng_draw = _draw_rng()
        assert actual_rng_draw[:2] == expected_rng_draw[:2]
        torch.testing.assert_close(actual_rng_draw[2], expected_rng_draw[2], rtol=0, atol=0)
        torch.testing.assert_close(actual_rng_draw[3], expected_rng_draw[3], rtol=0, atol=0)

        _optimizer_step(new_model, new_optimizer, step_index=1)
        _optimizer_step(expected_model, expected_optimizer, step_index=1)
        new_scheduler.step()
        expected_scheduler.step()
        _assert_model_matches(new_model, expected_model)
        _assert_optimizer_matches(new_model, new_optimizer, expected_model, expected_optimizer)
        assert new_scheduler.state_dict() == expected_scheduler.state_dict()
        assert new_scheduler.get_last_lr() == expected_scheduler.get_last_lr()

    dist.barrier()
    if rank == 0:
        print(f"FSDP2 GPU-direct training-state resize {old_world_size}->{new_world_size} passed")
    if rank in new_ranks:
        dist.destroy_process_group(new_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
