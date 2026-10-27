# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

import pytest

from verl.workers.rollout.llm_server import LLMServerManager


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    async def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeLoadBalancer:
    def __init__(self):
        self.added = []
        self.removed = []
        self.add_servers = _RemoteMethod(self.added.append)
        self.remove_servers = _RemoteMethod(self.removed.append)


class _FakeWorkerGroup:
    def __init__(self, workers):
        self.workers = list(workers)
        self.world_size = len(self.workers)


class _FakeReplica:
    instances = []

    def __init__(self, replica_rank, config, model_config, gpus_per_node):
        del model_config, gpus_per_node
        self.replica_rank = replica_rank
        self.world_size = config.tensor_model_parallel_size
        self.worker_group_rank = replica_rank
        self.workers = []
        self._server_address = f"server-{replica_rank}"
        self._server_handle = f"handle-{replica_rank}"
        self.park_count = 0
        self.wake_count = 0
        self.init_count = 0
        self.__class__.instances.append(self)

    async def init_hybrid(self, worker_group):
        begin = self.worker_group_rank * self.world_size
        self.workers = worker_group.workers[begin : begin + self.world_size]
        self.init_count += 1

    async def park(self):
        self.park_count += 1

    async def wake_up(self):
        self.wake_count += 1


def _replica(rank, workers, config):
    replica = _FakeReplica(rank, config, None, 8)
    replica.workers = list(workers)
    return replica


def _manager(active_replicas, worker_group, config):
    manager = LLMServerManager.__new__(LLMServerManager)
    manager.worker_group = worker_group
    manager.rollout_config = config
    manager.model_config = SimpleNamespace()
    manager.rollout_replica_class = _FakeReplica
    manager.rollout_replicas = list(active_replicas)
    manager.server_handles = [replica._server_handle for replica in active_replicas]
    manager.server_addresses = [replica._server_address for replica in active_replicas]
    manager.start_rank = min(replica.replica_rank for replica in active_replicas)
    manager._standby_replicas = {}
    manager._pending_activation_replicas = []
    manager.global_load_balancer = _FakeLoadBalancer()
    return manager


@pytest.fixture
def rollout_config():
    _FakeReplica.instances = []
    return SimpleNamespace(
        name="vllm",
        tensor_model_parallel_size=2,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        n_gpus_per_node=8,
    )


@pytest.mark.asyncio
async def test_hybrid_resize_reuses_parked_replica_on_round_trip(rollout_config):
    workers = [f"worker-{rank}" for rank in range(8)]
    replica_2 = _replica(2, workers[4:6], rollout_config)
    replica_3 = _replica(3, workers[6:8], rollout_config)
    manager = _manager([replica_2, replica_3], _FakeWorkerGroup(workers[4:8]), rollout_config)

    shrink_group = _FakeWorkerGroup(workers[6:8])
    await manager.prepare_hybrid_runtime_resize(shrink_group, start_rank=3)
    await manager.rebuild_hybrid_runtime(shrink_group, start_rank=3)

    assert manager.rollout_replicas == [replica_3]
    assert manager._standby_replicas == {2: replica_2}
    assert replica_2.park_count == 1
    assert manager.global_load_balancer.removed == [["server-2"]]

    expand_group = _FakeWorkerGroup(workers[4:8])
    await manager.prepare_hybrid_runtime_resize(expand_group, start_rank=2)
    await manager.rebuild_hybrid_runtime(expand_group, start_rank=2)

    assert manager.rollout_replicas == [replica_2, replica_3]
    assert manager._standby_replicas == {}
    assert manager._pending_activation_replicas == [replica_2]
    assert len(_FakeReplica.instances) == 2

    await manager.activate_pending_hybrid_replicas()
    assert manager.global_load_balancer.added == [{"server-2": "handle-2"}]
    assert manager._pending_activation_replicas == []


@pytest.mark.asyncio
async def test_hybrid_resize_launches_only_missing_replica(rollout_config):
    workers = [f"worker-{rank}" for rank in range(8)]
    replica_3 = _replica(3, workers[6:8], rollout_config)
    manager = _manager([replica_3], _FakeWorkerGroup(workers[6:8]), rollout_config)

    expand_group = _FakeWorkerGroup(workers[4:8])
    await manager.prepare_hybrid_runtime_resize(expand_group, start_rank=2)
    await manager.rebuild_hybrid_runtime(expand_group, start_rank=2)

    replica_2 = next(replica for replica in manager.rollout_replicas if replica.replica_rank == 2)
    assert replica_2 is not replica_3
    assert replica_2.workers == workers[4:6]
    assert replica_2.init_count == 1
    assert replica_3.workers == workers[6:8]
    assert manager._pending_activation_replicas == [replica_2]
    assert len(_FakeReplica.instances) == 2
