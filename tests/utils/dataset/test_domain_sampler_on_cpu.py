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

import pytest
from torch.utils.data import Dataset

from verl.utils.dataset.domain_sampler import DomainRoundRobinSampler, collect_domain_indices


class _FakeRLDataset(Dataset):
    """Minimal stand-in for `RLHFDataset`: a dataframe plus indexable rows."""

    def __init__(self, data_sources):
        import datasets

        self.dataframe = datasets.Dataset.from_dict({"data_source": list(data_sources)})

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        return {"data_source": self.dataframe[item]["data_source"]}


def _dataset(sizes=(("math", 6), ("code", 4), ("agent", 3))):
    sources = []
    for name, size in sizes:
        sources.extend([name] * size)
    return _FakeRLDataset(sources)


def _batches(sampler, batch_size):
    """Drain the sampler and split the flat index stream into batches."""
    indices = list(sampler)
    return [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]


def test_collect_domain_indices_groups_rows_by_domain():
    dataset = _dataset()
    grouped = collect_domain_indices(dataset, "data_source")

    assert list(grouped) == ["agent", "code", "math"]
    assert grouped["math"].tolist() == [0, 1, 2, 3, 4, 5]
    assert grouped["code"].tolist() == [6, 7, 8, 9]
    assert grouped["agent"].tolist() == [10, 11, 12]


def test_collect_domain_indices_rejects_missing_column():
    dataset = _dataset()
    with pytest.raises(ValueError, match="data_source_absent"):
        collect_domain_indices(dataset, "data_source_absent")

    class _NoDataframe(Dataset):
        def __len__(self):
            return 0

        def __getitem__(self, item):
            raise IndexError

    with pytest.raises(ValueError):
        collect_domain_indices(_NoDataframe(), "data_source")


def test_each_batch_is_single_domain_and_domains_rotate():
    dataset = _dataset()
    sampler = DomainRoundRobinSampler(dataset, batch_size=2, shuffle=False)

    domains_per_batch = []
    for batch in _batches(sampler, 2):
        domains = {dataset.dataframe[i]["data_source"] for i in batch}
        assert len(domains) == 1, f"batch mixes domains: {domains}"
        domains_per_batch.append(domains.pop())

    # agent has 3 samples, so it drops out after its first batch; code contributes
    # two batches and math three, all still rotating.
    assert domains_per_batch == ["agent", "code", "math", "code", "math", "math"]


def test_length_and_batch_count_drop_incomplete_domain_batches():
    dataset = _dataset()  # math 6, code 4, agent 3 -> agent can fill only one batch of 2
    sampler = DomainRoundRobinSampler(dataset, batch_size=2, shuffle=False)

    assert len(sampler) == 6 + 4 + 2
    assert len(_batches(sampler, 2)) == 6


def test_domain_smaller_than_batch_size_is_skipped():
    dataset = _dataset(sizes=(("math", 4), ("agent", 1)))
    sampler = DomainRoundRobinSampler(dataset, batch_size=2, shuffle=False)

    batches = _batches(sampler, 2)
    assert len(batches) == 2
    for batch in batches:
        assert {dataset.dataframe[i]["data_source"] for i in batch} == {"math"}


def test_shuffle_keeps_grouping_and_is_seed_reproducible():
    dataset = _dataset(sizes=(("math", 30), ("code", 20)))
    ordered = DomainRoundRobinSampler(dataset, batch_size=5, shuffle=False)
    shuffled = DomainRoundRobinSampler(dataset, batch_size=5, shuffle=True, seed=7)
    shuffled_again = DomainRoundRobinSampler(dataset, batch_size=5, shuffle=True, seed=7)

    # Unshuffled: every index is used once, each domain in file order, but the
    # domains are interleaved one batch at a time.
    assert sorted(ordered) == list(range(50))

    shuffled_indices = list(shuffled)
    assert shuffled_indices != list(ordered)
    assert shuffled_indices == list(shuffled_again)

    for batch in _batches(shuffled, 5):
        assert len({dataset.dataframe[i]["data_source"] for i in batch}) == 1


def test_epoch_rollover_reshuffles_and_covers_all_domains():
    dataset = _dataset(sizes=(("math", 4), ("code", 4)))
    sampler = DomainRoundRobinSampler(dataset, batch_size=2, shuffle=True, seed=3)

    first_epoch = list(sampler)
    second_epoch = list(sampler)

    assert len(first_epoch) == len(second_epoch) == 8
    # Same domains, different within-domain order across epochs.
    assert sorted(first_epoch) == sorted(second_epoch)
    assert first_epoch != second_epoch


def test_state_dict_resumes_mid_epoch():
    dataset = _dataset(sizes=(("math", 10), ("code", 10)))
    sampler = DomainRoundRobinSampler(dataset, batch_size=2, shuffle=True, seed=11)

    full_epoch = list(sampler)
    stop_at = 5

    resumed = DomainRoundRobinSampler(dataset, batch_size=2, shuffle=True, seed=11)
    consumed = [next(iter(resumed)) for _ in range(stop_at)]
    state = resumed.state_dict()
    assert state["pos"] == stop_at

    restored = DomainRoundRobinSampler(dataset, batch_size=2, shuffle=True, seed=99)
    restored.load_state_dict(state)
    assert consumed + list(restored) == full_epoch


def test_state_dict_restore_without_iter():
    """The dataloader restores the iterator after `iter()`, which must not lose the cursor."""
    dataset = _dataset(sizes=(("math", 10), ("code", 10)))
    sampler = DomainRoundRobinSampler(dataset, batch_size=2, shuffle=True, seed=5)

    expected = list(sampler)
    sampler.load_state_dict({"epoch": 0, "pos": 6, "seed": 5})
    iter(sampler)  # StatefulDataLoader re-creates the iterator after restoring
    assert list(sampler) == expected[6:]


def test_stateful_dataloader_resumes_on_the_same_domain_boundary():
    import io

    import torch
    from torchdata.stateful_dataloader import StatefulDataLoader

    dataset = _dataset(sizes=(("math", 12), ("code", 8)))

    def collate(rows):
        return {"data_source": [row["data_source"] for row in rows]}

    def build():
        return StatefulDataLoader(
            dataset=dataset,
            batch_size=4,
            drop_last=True,
            collate_fn=collate,
            sampler=DomainRoundRobinSampler(dataset, batch_size=4, shuffle=True, seed=17),
        )

    loader = build()
    full_epoch = [batch["data_source"][0] for batch in loader]

    loader = build()
    consumed = []
    for i, batch in enumerate(loader):
        consumed.append(batch["data_source"][0])
        if i == 1:
            buffer = io.BytesIO()
            torch.save(loader.state_dict(), buffer)
            state = torch.load(io.BytesIO(buffer.getvalue()), weights_only=False)
            break

    restored = build()
    restored.load_state_dict(state)
    assert consumed + [batch["data_source"][0] for batch in restored] == full_epoch


def test_invalid_batch_size_rejected():
    dataset = _dataset()
    with pytest.raises(ValueError, match="batch_size"):
        DomainRoundRobinSampler(dataset, batch_size=0)


def test_create_rl_sampler_selects_domain_sampler():
    from omegaconf import OmegaConf

    from verl.trainer.main_ppo import create_rl_sampler

    dataset = _dataset()
    config = OmegaConf.create({"shuffle": True, "train_batch_size": 2, "seed": 1})
    assert not isinstance(create_rl_sampler(config, dataset), DomainRoundRobinSampler)

    config.domain_batching = True
    sampler = create_rl_sampler(config, dataset)
    assert isinstance(sampler, DomainRoundRobinSampler)
    assert sampler.batch_size == 2

    config.gen_batch_size = 8
    assert create_rl_sampler(config, dataset).batch_size == 8
