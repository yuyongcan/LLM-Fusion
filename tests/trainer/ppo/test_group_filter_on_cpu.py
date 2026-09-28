# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Tests for DAPO dynamic sampling's group filter in verl.trainer.ppo.ray_trainer.
"""

import unittest

import numpy as np

from verl.trainer.ppo.ray_trainer import compute_group_keep_mask


class TestComputeGroupKeepMask(unittest.TestCase):
    def test_mixed_group_is_kept(self):
        scores = np.array([1.0, 0.0, 1.0, 0.0])
        uids = np.array(["a", "a", "a", "a"])
        keep, stats = compute_group_keep_mask(scores, uids)
        self.assertTrue(keep.all())
        self.assertEqual(stats["groups_total"], 1)
        self.assertEqual(stats["groups_kept"], 1)

    def test_all_wrong_group_is_dropped(self):
        scores = np.zeros(4)
        uids = np.array(["a"] * 4)
        keep, stats = compute_group_keep_mask(scores, uids)
        self.assertFalse(keep.any())
        self.assertEqual(stats["groups_all_wrong"], 1)
        self.assertEqual(stats["groups_all_right"], 0)
        self.assertEqual(stats["groups_kept"], 0)

    def test_all_right_group_is_dropped(self):
        scores = np.ones(4)
        uids = np.array(["a"] * 4)
        keep, stats = compute_group_keep_mask(scores, uids)
        self.assertFalse(keep.any())
        self.assertEqual(stats["groups_all_right"], 1)
        self.assertEqual(stats["groups_all_wrong"], 0)

    def test_all_right_when_no_group_scores_zero(self):
        """An all-right group must be labelled as such even when nothing in the
        batch scored zero -- the split is against 0.0, not the batch minimum."""
        scores = np.array([1.0, 1.0, 0.5, 0.5])
        uids = np.array(["a", "a", "b", "b"])
        _, stats = compute_group_keep_mask(scores, uids)
        self.assertEqual(stats["groups_all_right"], 2)
        self.assertEqual(stats["groups_all_wrong"], 0)

    def test_groups_are_kept_or_dropped_whole(self):
        """The mask must never split a group: partial groups would be normalized
        against a truncated baseline and silently yield a zero advantage."""
        rng = np.random.default_rng(0)
        n_groups, group_size = 40, 8
        uids = np.repeat([f"g{i}" for i in range(n_groups)], group_size)
        scores = rng.integers(0, 2, size=n_groups * group_size).astype(float)
        keep, stats = compute_group_keep_mask(scores, uids)
        for i in range(n_groups):
            rows = keep[i * group_size : (i + 1) * group_size]
            self.assertIn(rows.sum(), (0, group_size), f"group g{i} was split")
        self.assertEqual(stats["groups_total"], n_groups)
        self.assertEqual(int(keep.sum()) // group_size, stats["groups_kept"])

    def test_stats_partition_every_group(self):
        scores = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        uids = np.array(["a", "a", "b", "b", "c", "c"])
        keep, stats = compute_group_keep_mask(scores, uids)
        self.assertEqual(
            stats["groups_kept"] + stats["groups_all_wrong"] + stats["groups_all_right"],
            stats["groups_total"],
        )
        self.assertEqual(stats["groups_total"], 3)
        np.testing.assert_array_equal(keep, [True, True, False, False, False, False])

    def test_non_contiguous_groups(self):
        """Rows of one group need not be adjacent -- grouping is by uid, not position."""
        scores = np.array([1.0, 1.0, 0.0, 1.0])
        uids = np.array(["a", "b", "a", "b"])
        keep, stats = compute_group_keep_mask(scores, uids)
        np.testing.assert_array_equal(keep, [True, False, True, False])
        self.assertEqual(stats["groups_kept"], 1)
        self.assertEqual(stats["groups_all_right"], 1)

    def test_single_sample_group_is_dropped(self):
        """A group of one has no spread, so it carries no gradient."""
        scores = np.array([1.0])
        uids = np.array(["a"])
        keep, stats = compute_group_keep_mask(scores, uids)
        self.assertFalse(keep.any())
        self.assertEqual(stats["groups_total"], 1)

    def test_empty_input(self):
        keep, stats = compute_group_keep_mask(np.array([]), np.array([]))
        self.assertEqual(len(keep), 0)
        self.assertEqual(stats["groups_total"], 0)
        self.assertEqual(stats["groups_kept"], 0)

    def test_non_binary_rewards(self):
        """Partial-credit scores must compare exactly, not via a std threshold."""
        scores = np.array([0.3, 0.7, 0.5, 0.5])
        uids = np.array(["a", "a", "b", "b"])
        keep, stats = compute_group_keep_mask(scores, uids)
        np.testing.assert_array_equal(keep, [True, True, False, False])
        self.assertEqual(stats["groups_kept"], 1)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            compute_group_keep_mask(np.zeros(4), np.array(["a", "a"]))

    def test_mask_is_boolean_for_select_idxs(self):
        """DataProto.select_idxs branches on dtype == bool to mean 'mask', so an
        integer mask would be read as positional indices instead."""
        keep, _ = compute_group_keep_mask(np.array([1.0, 0.0]), np.array(["a", "a"]))
        self.assertEqual(keep.dtype, np.dtype(bool))


if __name__ == "__main__":
    unittest.main()
