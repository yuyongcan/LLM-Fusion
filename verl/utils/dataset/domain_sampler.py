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

"""Domain-aware samplers for RL datasets.

Mixed-domain RL (e.g. the `Mix` config) loads one parquet blend whose rows carry a
`data_source` column, and the default sampler draws a shuffled batch from the whole
blend -- so every step trains on a mix of Math/Science/Code/IF/Agent prompts.

`DomainRoundRobinSampler` is the opt-in alternative: it hands out one single-domain
batch at a time and rotates over the domains, so step ``k`` trains on domain
``k % num_domains``. Enable it with ``data.domain_batching=True``.
"""

import logging
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


def collect_domain_indices(dataset: Dataset, domain_key: str = "data_source") -> dict[str, np.ndarray]:
    """Group dataset row indices by the value of a domain column.

    Args:
        dataset: The RL dataset. It must expose the underlying `datasets.Dataset` as
            ``dataset.dataframe`` (as `RLHFDataset` does), so the whole domain column
            can be read without materialising every row.
        domain_key: Name of the column holding the domain tag (e.g. "data_source").

    Returns:
        Mapping from domain name to the row indices belonging to it, sorted by name.

    Raises:
        ValueError: If the dataset has no readable dataframe or no such column.
    """
    dataframe = getattr(dataset, "dataframe", None)
    column_names = getattr(dataframe, "column_names", None)
    if dataframe is None or column_names is None or domain_key not in column_names:
        raise ValueError(
            f"Cannot group dataset by '{domain_key}': expected `dataset.dataframe` to be a "
            f"datasets.Dataset exposing that column (got {type(dataframe).__name__}). "
            "Set `data.domain_key` to an existing column, or disable `data.domain_batching`."
        )

    grouped: dict[str, list[int]] = {}
    for index, value in enumerate(dataframe[domain_key]):
        grouped.setdefault(str(value), []).append(index)

    return {name: np.asarray(indices, dtype=np.int64) for name, indices in sorted(grouped.items())}


class DomainRoundRobinSampler(Sampler[int]):
    """Sampler that yields one single-domain batch per domain, in rotation.

    Instead of blending every domain into every batch, the sampler walks the domains
    in a fixed (sorted) order and takes `batch_size` samples from one domain at a
    time: batch 0 comes from domain 0, batch 1 from domain 1, and so on, wrapping
    around until the domains run out. Within a domain the samples follow
    `data.shuffle` -- a per-epoch permutation when shuffling is on, otherwise file
    order. A domain whose remaining samples cannot fill a whole batch is skipped,
    mirroring the `drop_last=True` the RL dataloader already uses.

    The sampler is stateful (`state_dict`/`load_state_dict`) so
    `StatefulDataLoader` can checkpoint and restore the position mid-epoch.

    Args:
        dataset: The RL dataset to sample from.
        batch_size: Number of samples per batch. Must match the dataloader's
            `batch_size`, i.e. `data.gen_batch_size or data.train_batch_size`.
        domain_key: Column that identifies a sample's domain.
        shuffle: Whether to permute each domain's samples every epoch.
        seed: Seed for the permutations. A random one is drawn when None.

    Raises:
        ValueError: If `batch_size` is not positive or the domain column is missing.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        domain_key: str = "data_source",
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        if batch_size is None or int(batch_size) <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.domain_key = domain_key
        self.shuffle = bool(shuffle)
        self.seed = int(seed) if seed is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())

        self.domain_indices = collect_domain_indices(dataset, domain_key)
        self.domain_order = sorted(self.domain_indices)

        self._epoch = 0
        self._pos = 0
        self._order: Optional[np.ndarray] = None

        self._log_plan()

    @property
    def domains(self) -> list[str]:
        """The domains covered by this sampler, in rotation order."""
        return list(self.domain_order)

    def _log_plan(self) -> None:
        """Report how many batches each domain contributes per epoch."""
        plan = {name: len(indices) // self.batch_size for name, indices in self.domain_indices.items()}
        skipped = [name for name, batches in plan.items() if batches == 0]
        if skipped:
            logger.warning(
                "domain_batching: %d domain(s) cannot fill a %d-sample batch and will never be "
                "sampled: %s. Lower data.train_batch_size or merge them into another domain.",
                len(skipped),
                self.batch_size,
                ", ".join(skipped),
            )
        logger.info(
            "domain_batching: %d domain(s), ~%d batch(es) per epoch, batches per domain: %s",
            len(self.domain_order),
            len(self) // self.batch_size,
            plan,
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to derive the per-domain permutations."""
        self._epoch = int(epoch)
        self._order = None

    def _permuted(self, slot: int, indices: np.ndarray) -> np.ndarray:
        """Return this epoch's sample order for a domain."""
        if not self.shuffle:
            return indices
        # Seeded per (run, epoch, domain) so a resumed run replays the same order
        # without having to checkpoint the permutation itself.
        rng = np.random.default_rng([self.seed, self._epoch, slot])
        return indices[rng.permutation(len(indices))]

    def _build_order(self) -> np.ndarray:
        """Interleave the domains one `batch_size` chunk at a time."""
        per_domain = [self._permuted(slot, self.domain_indices[name]) for slot, name in enumerate(self.domain_order)]
        cursors = [0] * len(per_domain)
        done = [len(indices) < self.batch_size for indices in per_domain]

        chunks: list[np.ndarray] = []
        while not all(done):
            for slot, indices in enumerate(per_domain):
                if done[slot]:
                    continue
                start = cursors[slot]
                chunks.append(indices[start : start + self.batch_size])
                cursors[slot] = start + self.batch_size
                if cursors[slot] + self.batch_size > len(indices):
                    # Whatever is left cannot fill another batch.
                    done[slot] = True

        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(chunks)

    def __iter__(self) -> "DomainRoundRobinSampler":
        """Return the sampler iterator, rolling over to the next epoch when spent."""
        if self._order is not None and self._pos >= len(self._order):
            self._epoch += 1
            self._pos = 0
            self._order = None
        if self._order is None:
            self._order = self._build_order()
        return self

    def __next__(self) -> int:
        """Return the next sample index."""
        if self._order is None:
            # iter() was not called first (or the state was just restored).
            self._order = self._build_order()
        if self._pos >= len(self._order):
            raise StopIteration
        index = int(self._order[self._pos])
        self._pos += 1
        return index

    def __len__(self) -> int:
        """Number of sampled indices per epoch: the full batches every domain can fill."""
        return sum(len(indices) // self.batch_size * self.batch_size for indices in self.domain_indices.values())

    def state_dict(self) -> dict[str, Any]:
        """Return the resumable state: epoch, position in the epoch and the seed."""
        return {"epoch": int(self._epoch), "pos": int(self._pos), "seed": int(self.seed)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore a state produced by `state_dict`.

        Args:
            state_dict: Mapping with "epoch", "pos" and (optionally) "seed".
        """
        self._epoch = int(state_dict["epoch"])
        self._pos = int(state_dict["pos"])
        if "seed" in state_dict:
            self.seed = int(state_dict["seed"])
        # Rebuilt lazily: the order is a pure function of (seed, epoch), so the
        # restored position stays valid.
        self._order = None
