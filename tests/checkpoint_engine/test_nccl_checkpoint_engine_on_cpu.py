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

from verl.checkpoint_engine import nccl_checkpoint_engine


def test_finalize_accepts_already_finalized_non_member(monkeypatch):
    engine = nccl_checkpoint_engine.NCCLCheckpointEngine.__new__(nccl_checkpoint_engine.NCCLCheckpointEngine)
    engine.rebuild_group = True
    engine.rank = None
    engine.world_size = None
    engine.group_name = "resize"
    engine.send_buf = object()
    engine.recv_buf = object()

    destroyed = []
    monkeypatch.setattr(
        nccl_checkpoint_engine.collective,
        "destroy_collective_group",
        destroyed.append,
    )
    monkeypatch.setattr(nccl_checkpoint_engine.torch.cuda, "empty_cache", lambda: None)

    engine.finalize()

    assert destroyed == []
    assert engine.send_buf is None
    assert engine.recv_buf is None
