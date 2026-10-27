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

"""Exercise repeated FSDP2 subgroup teardown/rebuild in one process lifetime."""

import gc
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard

from verl.utils.fsdp_utils import fsdp2_load_full_state_dict
from verl.workers.engine.fsdp.utils import create_device_mesh, release_device_mesh


def _build_and_load(group) -> tuple[nn.Module, object]:
    mesh = create_device_mesh(dist.get_world_size(group), fsdp_size=-1, process_group=group)
    torch.manual_seed(1234)
    model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4)).cuda()
    full_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    fully_shard(model, mesh=mesh)
    fsdp2_load_full_state_dict(model, full_state, mesh, cpu_offload=None)
    return model, mesh


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == 8

    sizes = tuple(int(value) for value in os.environ.get("RESIZE_SIZES", "4,6,4").split(","))
    for size in sizes:
        ranks = list(range(size))
        groups = [dist.new_group(ranks=ranks, group_desc=f"resize_{size}_{index}") for index in range(3)]
        model, mesh = _build_and_load(groups[0]) if rank in ranks else (None, None)
        dist.barrier()

        del model
        release_device_mesh(mesh)
        gc.collect()
        torch.cuda.empty_cache()
        for group in groups:
            dist.barrier()
            if rank in ranks:
                dist.destroy_process_group(group)
            dist.barrier()

    if rank == 0:
        print("FSDP2 process-group round trip 4->6->4 passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
