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

"""GPU-resident FSDP2 model-state transfer helpers for dynamic resize."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import prod
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

from verl.utils.device import get_torch_device


def compute_1d_shard_range(dim_size: int, world_size: int, rank: int) -> tuple[int, int]:
    """Return the offset and size used by a one-dimensional DTensor ``Shard(0)``."""
    if dim_size < 0:
        raise ValueError(f"dim_size must be non-negative, got {dim_size}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    shard_size, offset = Shard._local_shard_size_and_offset(dim_size, world_size, rank)
    return int(offset), int(shard_size)


@dataclass(frozen=True, slots=True)
class ShardInterval:
    offset: int
    size: int

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass(frozen=True, slots=True)
class ShardOverlap:
    source_rank: int
    target_rank: int
    global_offset: int
    source_offset: int
    target_offset: int
    size: int


class FSDPResizeOverlapPlanner:
    """Plan the interval intersections between old and new one-dimensional shards."""

    def __init__(self, dim_size: int, old_world_size: int, new_world_size: int):
        if old_world_size <= 0 or new_world_size <= 0:
            raise ValueError(
                f"old_world_size and new_world_size must be positive, got {old_world_size}, {new_world_size}"
            )
        self.dim_size = dim_size
        self.old_world_size = old_world_size
        self.new_world_size = new_world_size
        self.source_intervals = [
            ShardInterval(*compute_1d_shard_range(dim_size, old_world_size, rank)) for rank in range(old_world_size)
        ]
        self.target_intervals = [
            ShardInterval(*compute_1d_shard_range(dim_size, new_world_size, rank)) for rank in range(new_world_size)
        ]

    def overlap(self, source_rank: int, target_rank: int) -> ShardOverlap:
        source = self.source_intervals[source_rank]
        target = self.target_intervals[target_rank]
        start = max(source.offset, target.offset)
        end = min(source.end, target.end)
        size = max(end - start, 0)
        return ShardOverlap(
            source_rank=source_rank,
            target_rank=target_rank,
            global_offset=start,
            source_offset=start - source.offset if size else 0,
            target_offset=start - target.offset if size else 0,
            size=size,
        )

    def send_row_splits(self, source_rank: int) -> list[int]:
        return [self.overlap(source_rank, target_rank).size for target_rank in range(self.new_world_size)]

    def recv_row_splits(self, target_rank: int) -> list[int]:
        return [self.overlap(source_rank, target_rank).size for source_rank in range(self.old_world_size)]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _resolve_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported tensor dtype in resize manifest: {name}")
    return dtype


def _build_manifest(model: torch.nn.Module) -> list[dict[str, Any]]:
    manifest = []
    for name, value in sorted(model.state_dict().items()):
        if not isinstance(value, torch.Tensor):
            continue
        sharded = False
        shard_dim = None
        if isinstance(value, DTensor):
            placements = value.placements
            if len(placements) != 1:
                raise NotImplementedError(
                    f"GPU-direct resize only supports a one-dimensional placement, got {placements} for {name}"
                )
            placement = placements[0]
            if isinstance(placement, Shard):
                sharded = True
                shard_dim = int(placement.dim)
            elif not isinstance(placement, Replicate):
                raise NotImplementedError(
                    f"GPU-direct resize only supports Shard or Replicate placement, got {placements} for {name}"
                )
            if sharded and shard_dim != 0:
                raise NotImplementedError(
                    f"GPU-direct resize only supports Shard(0), got Shard({shard_dim}) for {name}"
                )
        manifest.append(
            {
                "name": name,
                "shape": tuple(int(v) for v in value.shape),
                "dtype": _dtype_name(value.dtype),
                "sharded": sharded,
                "shard_dim": shard_dim,
            }
        )
    return manifest


def _optimizer_param_groups(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, Any]]:
    name_by_parameter_id = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        parameter_names = []
        for parameter in group["params"]:
            name = name_by_parameter_id.get(id(parameter))
            if name is None:
                raise RuntimeError(f"optimizer parameter group {index} contains a parameter absent from the model")
            parameter_names.append(name)
        options = {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
        if any(isinstance(value, torch.Tensor) for value in options.values()):
            raise NotImplementedError("GPU-direct optimizer resize does not support tensor-valued param-group options")
        groups.append({"parameter_names": parameter_names, "options": options})
    return groups


def _build_optimizer_manifest(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], torch.Tensor]]:
    name_by_parameter_id = {id(parameter): name for name, parameter in model.named_parameters()}
    manifest = []
    tensor_state: dict[tuple[str, str], torch.Tensor] = {}
    for parameter, state in optimizer.state.items():
        parameter_name = name_by_parameter_id.get(id(parameter))
        if parameter_name is None:
            raise RuntimeError("optimizer state contains a parameter absent from the model")
        for state_name, value in state.items():
            item: dict[str, Any] = {"parameter_name": parameter_name, "state_name": str(state_name)}
            key = (parameter_name, str(state_name))
            if isinstance(value, DTensor):
                placements = value.placements
                if len(placements) != 1 or not isinstance(placements[0], Shard) or placements[0].dim != 0:
                    raise NotImplementedError(
                        "GPU-direct optimizer resize only supports one-dimensional Shard(0) DTensor state, "
                        f"got {placements} for {parameter_name}.{state_name}"
                    )
                item.update(
                    {
                        "kind": "sharded_tensor",
                        "shape": tuple(int(v) for v in value.shape),
                        "dtype": _dtype_name(value.dtype),
                    }
                )
                tensor_state[key] = value
            elif isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise NotImplementedError(
                        "GPU-direct optimizer resize only supports DTensor shards and scalar plain tensors, "
                        f"got shape {tuple(value.shape)} for {parameter_name}.{state_name}"
                    )
                item.update(
                    {
                        "kind": "scalar_tensor",
                        "shape": tuple(int(v) for v in value.shape),
                        "dtype": _dtype_name(value.dtype),
                        "device_type": value.device.type,
                        "value": value.item(),
                    }
                )
            elif isinstance(value, bool | int | float | str) or value is None:
                item.update({"kind": "value", "value": copy.deepcopy(value)})
            else:
                raise NotImplementedError(
                    f"GPU-direct optimizer resize does not support state type {type(value)} "
                    f"for {parameter_name}.{state_name}"
                )
            manifest.append(item)
    manifest.sort(key=lambda item: (item["parameter_name"], item["state_name"]))
    return manifest, tensor_state


class FSDPResizeStateTransferManager:
    """Transfer FSDP2 local shards directly between differently sized meshes."""

    def __init__(
        self,
        *,
        old_ranks: list[int],
        new_ranks: list[int],
        transfer_group: dist.ProcessGroup,
        device: torch.device,
    ):
        if not old_ranks or not new_ranks:
            raise ValueError("old_ranks and new_ranks must not be empty")
        if old_ranks != list(range(len(old_ranks))) or new_ranks != list(range(len(new_ranks))):
            raise NotImplementedError("FSDP resize transfer currently requires contiguous ranks starting at zero")
        self.old_ranks = old_ranks
        self.new_ranks = new_ranks
        self.transfer_group = transfer_group
        self.device = device
        self.global_rank = dist.get_rank()
        self.source_rank = old_ranks[0]
        self.transfer_world_size = dist.get_world_size(group=transfer_group)
        expected_transfer_world_size = len(set(old_ranks) | set(new_ranks))
        if self.transfer_world_size != expected_transfer_world_size:
            raise ValueError(
                "transfer group size does not match the source/target rank union: "
                f"expected {expected_transfer_world_size}, got {self.transfer_world_size}"
            )

    @torch.no_grad()
    def stage(self, model: torch.nn.Module | None) -> tuple[dict[str, torch.Tensor] | None, dict[str, int]]:
        is_old_rank = self.global_rank in self.old_ranks
        is_new_rank = self.global_rank in self.new_ranks
        if is_old_rank and model is None:
            raise RuntimeError(f"old actor rank {self.global_rank} has no active FSDP model")

        source_state = model.state_dict() if is_old_rank else None
        object_list = [_build_manifest(model) if self.global_rank == self.source_rank else None]
        dist.broadcast_object_list(
            object_list,
            src=self.source_rank,
            group=self.transfer_group,
            device=self.device,
        )
        manifest = object_list[0]
        if not isinstance(manifest, list) or not manifest:
            raise RuntimeError("received an empty FSDP resize manifest")

        target_state: dict[str, torch.Tensor] | None = {} if is_new_rank else None
        staged_bytes = 0
        sharded_tensor_count = 0
        replicated_tensor_count = 0
        for item in manifest:
            name = str(item["name"])
            if bool(item["sharded"]):
                local_tensor = self._transfer_sharded_tensor(name, item, source_state)
                sharded_tensor_count += 1
            else:
                local_tensor = self._transfer_replicated_tensor(name, item, source_state)
                replicated_tensor_count += 1
            if is_new_rank:
                if local_tensor is None:
                    raise RuntimeError(f"target actor rank {self.global_rank} did not receive tensor {name}")
                assert target_state is not None
                target_state[name] = local_tensor
                staged_bytes += local_tensor.numel() * local_tensor.element_size()

        get_torch_device().synchronize(self.device)
        return target_state, {
            "tensor_count": len(manifest),
            "sharded_tensor_count": sharded_tensor_count,
            "replicated_tensor_count": replicated_tensor_count,
            "staged_bytes": staged_bytes,
        }

    @torch.no_grad()
    def stage_optimizer(
        self,
        model: torch.nn.Module | None,
        optimizer: torch.optim.Optimizer | None,
    ) -> tuple[dict[str, Any] | None, dict[str, int]]:
        """Stage name-keyed optimizer shards for the target FSDP mesh."""
        is_old_rank = self.global_rank in self.old_ranks
        is_new_rank = self.global_rank in self.new_ranks
        if is_old_rank and (model is None or optimizer is None):
            raise RuntimeError(f"old actor rank {self.global_rank} has no active model optimizer")

        source_state = None
        local_manifest = None
        local_param_groups = None
        if is_old_rank:
            assert model is not None and optimizer is not None
            local_manifest, source_state = _build_optimizer_manifest(model, optimizer)
            local_param_groups = _optimizer_param_groups(model, optimizer)

        object_list = [
            {"manifest": local_manifest, "param_groups": local_param_groups}
            if self.global_rank == self.source_rank
            else None
        ]
        dist.broadcast_object_list(
            object_list,
            src=self.source_rank,
            group=self.transfer_group,
            device=self.device,
        )
        envelope = object_list[0]
        if not isinstance(envelope, dict):
            raise RuntimeError("received an invalid FSDP optimizer resize manifest")
        manifest = envelope.get("manifest")
        param_groups = envelope.get("param_groups")
        if not isinstance(manifest, list) or not isinstance(param_groups, list):
            raise RuntimeError("received an invalid FSDP optimizer resize manifest")

        target_state: dict[str, Any] | None = None
        if is_new_rank:
            target_state = {"manifest": manifest, "param_groups": param_groups, "state": {}}
        staged_bytes = 0
        sharded_tensor_count = 0
        scalar_state_count = 0
        for item in manifest:
            parameter_name = str(item["parameter_name"])
            state_name = str(item["state_name"])
            key = (parameter_name, state_name)
            kind = str(item["kind"])
            if kind == "sharded_tensor":
                local_value = self._transfer_sharded_tensor(
                    f"{parameter_name}.{state_name}",
                    item,
                    source_state,
                    source_key=key,
                )
                sharded_tensor_count += 1
            elif kind in {"scalar_tensor", "value"}:
                local_value = copy.deepcopy(item.get("value"))
                scalar_state_count += 1
            else:
                raise RuntimeError(f"unsupported optimizer resize state kind: {kind}")

            if is_new_rank:
                assert target_state is not None
                target_state["state"][key] = local_value
                if isinstance(local_value, torch.Tensor):
                    staged_bytes += local_value.numel() * local_value.element_size()
                elif kind == "scalar_tensor":
                    staged_bytes += torch.empty((), dtype=_resolve_dtype(str(item["dtype"]))).element_size()

        get_torch_device().synchronize(self.device)
        return target_state, {
            "tensor_count": len(manifest),
            "sharded_tensor_count": sharded_tensor_count,
            "scalar_state_count": scalar_state_count,
            "staged_bytes": staged_bytes,
        }

    def _transfer_sharded_tensor(
        self,
        name: str,
        item: dict[str, Any],
        source_state: dict[Any, torch.Tensor] | None,
        *,
        source_key: Any | None = None,
    ) -> torch.Tensor | None:
        shape = tuple(int(v) for v in item["shape"])
        if not shape:
            raise RuntimeError(f"cannot Shard(0) scalar tensor {name}")
        dtype = _resolve_dtype(str(item["dtype"]))
        planner = FSDPResizeOverlapPlanner(shape[0], len(self.old_ranks), len(self.new_ranks))
        row_width = prod(shape[1:])

        if self.global_rank in self.old_ranks:
            assert source_state is not None
            source_value = source_state.get(name if source_key is None else source_key)
            if not isinstance(source_value, DTensor):
                raise RuntimeError(f"source FSDP state tensor {name} is not a DTensor")
            source_local = source_value.to_local().detach().to(self.device, non_blocking=True)
            source_interval = planner.source_intervals[self.global_rank]
            expected_shape = (source_interval.size, *shape[1:])
            if tuple(source_local.shape) != expected_shape:
                raise RuntimeError(
                    f"source local shape mismatch for {name} on rank {self.global_rank}: "
                    f"expected {expected_shape}, got {tuple(source_local.shape)}"
                )
            send_row_splits = planner.send_row_splits(self.global_rank)
            send_chunks = []
            for target_rank, row_count in enumerate(send_row_splits):
                if row_count == 0:
                    continue
                overlap = planner.overlap(self.global_rank, target_rank)
                send_chunks.append(source_local.narrow(0, overlap.source_offset, overlap.size).contiguous().view(-1))
            send_tensor = torch.cat(send_chunks) if send_chunks else torch.empty(0, dtype=dtype, device=self.device)
        else:
            send_row_splits = [0] * len(self.new_ranks)
            send_tensor = torch.empty(0, dtype=dtype, device=self.device)

        send_row_splits.extend([0] * (self.transfer_world_size - len(send_row_splits)))
        is_new_rank = self.global_rank in self.new_ranks
        if is_new_rank:
            target_rank = self.new_ranks.index(self.global_rank)
            recv_row_splits = planner.recv_row_splits(target_rank)
        else:
            target_rank = None
            recv_row_splits = [0] * len(self.old_ranks)
        recv_row_splits.extend([0] * (self.transfer_world_size - len(recv_row_splits)))
        send_splits = [row_count * row_width for row_count in send_row_splits]
        recv_splits = [row_count * row_width for row_count in recv_row_splits]
        recv_tensor = torch.empty(sum(recv_splits), dtype=dtype, device=self.device)
        dist.all_to_all_single(
            recv_tensor,
            send_tensor,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.transfer_group,
        )

        if target_rank is None:
            return None
        target_interval = planner.target_intervals[target_rank]
        expected_numel = target_interval.size * row_width
        if recv_tensor.numel() != expected_numel:
            raise RuntimeError(
                f"target shard size mismatch for {name} on rank {self.global_rank}: "
                f"expected {expected_numel}, got {recv_tensor.numel()}"
            )
        return recv_tensor.reshape(target_interval.size, *shape[1:])

    def _transfer_replicated_tensor(
        self,
        name: str,
        item: dict[str, Any],
        source_state: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor | None:
        shape = tuple(int(v) for v in item["shape"])
        dtype = _resolve_dtype(str(item["dtype"]))
        if self.global_rank == self.source_rank:
            assert source_state is not None
            source_value = source_state.get(name)
            if not isinstance(source_value, torch.Tensor):
                raise RuntimeError(f"source replicated state tensor {name} is unavailable")
            source_local = source_value.to_local() if isinstance(source_value, DTensor) else source_value
            full_tensor = source_local.detach().to(self.device, non_blocking=True)
        else:
            full_tensor = torch.empty(shape, dtype=dtype, device=self.device)
        dist.broadcast(full_tensor, src=self.source_rank, group=self.transfer_group)
        return full_tensor.clone() if self.global_rank in self.new_ranks else None


@torch.no_grad()
def stage_fsdp2_model_state(
    model: torch.nn.Module | None,
    *,
    old_ranks: list[int],
    new_ranks: list[int],
    transfer_group: dist.ProcessGroup,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, int]]:
    """Stage target shards with direct local-shard variable all-to-all transfers."""
    manager = FSDPResizeStateTransferManager(
        old_ranks=old_ranks,
        new_ranks=new_ranks,
        transfer_group=transfer_group,
        device=device,
    )
    target_state, stats = manager.stage(model)
    return target_state, stats


@torch.no_grad()
def stage_fsdp2_optimizer_state(
    model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
    *,
    old_ranks: list[int],
    new_ranks: list[int],
    transfer_group: dist.ProcessGroup,
    device: torch.device,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """Stage name-keyed optimizer state with direct local-shard transfers."""
    manager = FSDPResizeStateTransferManager(
        old_ranks=old_ranks,
        new_ranks=new_ranks,
        transfer_group=transfer_group,
        device=device,
    )
    return manager.stage_optimizer(model, optimizer)


@torch.no_grad()
def restore_fsdp2_model_state(
    model: torch.nn.Module,
    staged_state: dict[str, torch.Tensor],
) -> dict[str, int]:
    """Copy previously staged target shards into a rebuilt FSDP2 model."""
    target_state = model.state_dict()
    missing = sorted(set(target_state) - set(staged_state))
    unexpected = sorted(set(staged_state) - set(target_state))
    if missing or unexpected:
        raise RuntimeError(
            f"FSDP resize state keys do not match rebuilt model: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    copy_plan: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    restored_bytes = 0
    for name, target_value in sorted(target_state.items()):
        staged_value = staged_state[name]
        target_local = target_value.to_local() if isinstance(target_value, DTensor) else target_value
        if tuple(target_local.shape) != tuple(staged_value.shape):
            raise RuntimeError(
                f"FSDP resize local shape mismatch for {name}: "
                f"target={tuple(target_local.shape)}, staged={tuple(staged_value.shape)}"
            )
        copy_plan.append((name, target_local, staged_value))
        restored_bytes += staged_value.numel() * staged_value.element_size()

    for _, target_local, staged_value in copy_plan:
        target_local.copy_(staged_value.to(device=target_local.device, dtype=target_local.dtype, non_blocking=True))

    get_torch_device().synchronize()
    return {"tensor_count": len(copy_plan), "restored_bytes": restored_bytes}


@torch.no_grad()
def restore_fsdp2_optimizer_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    staged_state: dict[str, Any],
) -> dict[str, int]:
    """Restore name-keyed optimizer shards into a rebuilt FSDP2 optimizer."""
    manifest = staged_state.get("manifest")
    staged_param_groups = staged_state.get("param_groups")
    state = staged_state.get("state")
    if not isinstance(manifest, list) or not isinstance(staged_param_groups, list) or not isinstance(state, dict):
        raise RuntimeError("invalid staged FSDP optimizer state")

    parameters_by_name = dict(model.named_parameters())
    name_by_parameter_id = {id(parameter): name for name, parameter in parameters_by_name.items()}
    if len(staged_param_groups) != len(optimizer.param_groups):
        raise RuntimeError(
            "optimizer parameter group count changed during resize: "
            f"staged={len(staged_param_groups)}, target={len(optimizer.param_groups)}"
        )
    for index, (staged_group, target_group) in enumerate(zip(staged_param_groups, optimizer.param_groups, strict=True)):
        target_names = [name_by_parameter_id.get(id(parameter)) for parameter in target_group["params"]]
        if target_names != staged_group.get("parameter_names"):
            raise RuntimeError(f"optimizer parameter group {index} changed during resize")
        target_group.update(copy.deepcopy(staged_group["options"]))

    expected_keys = {(str(item["parameter_name"]), str(item["state_name"])) for item in manifest}
    if set(state) != expected_keys:
        missing = sorted(expected_keys - set(state))
        unexpected = sorted(set(state) - expected_keys)
        raise RuntimeError(
            f"optimizer resize state keys do not match manifest: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    optimizer.state.clear()
    restored_bytes = 0
    for item in manifest:
        parameter_name = str(item["parameter_name"])
        state_name = str(item["state_name"])
        parameter = parameters_by_name.get(parameter_name)
        if parameter is None:
            raise RuntimeError(f"optimizer parameter {parameter_name} is absent from the rebuilt model")
        staged_value = state[(parameter_name, state_name)]
        kind = str(item["kind"])
        if kind == "sharded_tensor":
            if not isinstance(parameter, DTensor) or not isinstance(staged_value, torch.Tensor):
                raise RuntimeError(f"cannot restore sharded optimizer state {parameter_name}.{state_name}")
            target_value = torch.zeros_like(parameter, dtype=staged_value.dtype, requires_grad=False)
            if not isinstance(target_value, DTensor):
                raise RuntimeError(f"rebuilt optimizer state {parameter_name}.{state_name} is not a DTensor")
            target_local = target_value.to_local()
            if tuple(target_local.shape) != tuple(staged_value.shape):
                raise RuntimeError(
                    f"optimizer local shape mismatch for {parameter_name}.{state_name}: "
                    f"target={tuple(target_local.shape)}, staged={tuple(staged_value.shape)}"
                )
            target_local.copy_(staged_value.to(device=target_local.device, dtype=target_local.dtype))
            restored_value: Any = target_value
            restored_bytes += staged_value.numel() * staged_value.element_size()
        elif kind == "scalar_tensor":
            dtype = _resolve_dtype(str(item["dtype"]))
            device_type = str(item.get("device_type", "cpu"))
            device = torch.device("cpu") if device_type == "cpu" else parameter.to_local().device
            restored_value = torch.tensor(staged_value, dtype=dtype, device=device).reshape(tuple(item["shape"]))
            restored_bytes += restored_value.numel() * restored_value.element_size()
        elif kind == "value":
            restored_value = copy.deepcopy(staged_value)
        else:
            raise RuntimeError(f"unsupported optimizer resize state kind: {kind}")
        optimizer.state[parameter][state_name] = restored_value

    get_torch_device().synchronize()
    return {"tensor_count": len(manifest), "restored_bytes": restored_bytes}
