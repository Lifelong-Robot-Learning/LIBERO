"""Replay memory budgets cannot expose samples past a task's dataset."""
import unittest
import torch
from torch.utils.data import DataLoader, Dataset
from libero.lifelong.datasets import TruncatedSequenceDataset


class FiniteDataset(Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        return torch.tensor(index)


class TruncatedReplayDatasetTest(unittest.TestCase):
    def test_large_memory_budget_is_bounded_by_available_samples(self):
        dataset = TruncatedSequenceDataset(FiniteDataset(3), 10)
        self.assertEqual(len(dataset), 3)
        batches = list(DataLoader(dataset, batch_size=2, shuffle=False))
        self.assertEqual(torch.cat(batches).tolist(), [0, 1, 2])

    def test_small_memory_budget_is_preserved(self):
        dataset = TruncatedSequenceDataset(FiniteDataset(10), 3)
        self.assertEqual(len(dataset), 3)
        self.assertEqual([dataset[index].item() for index in range(len(dataset))], [0, 1, 2])

    def test_empty_task_and_zero_budget_are_empty(self):
        self.assertEqual(len(TruncatedSequenceDataset(FiniteDataset(0), 10)), 0)
        self.assertEqual(len(TruncatedSequenceDataset(FiniteDataset(10), 0)), 0)

    def test_negative_memory_budget_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            TruncatedSequenceDataset(FiniteDataset(10), -1)


if __name__ == "__main__":
    unittest.main()

