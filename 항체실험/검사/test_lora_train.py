import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import unittest

from LoRA import EarlyStopping, antibody_loss_sum, accumulation_windows


class TrainingTests(unittest.TestCase):
    def test_causal_mask_keeps_first_antibody_target(self):
        import torch
        logits = torch.zeros((1, 5, 3), requires_grad=True)
        labels = torch.tensor([[-100, -100, 2, 1, 0]])
        nll, count = antibody_loss_sum(logits, labels)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(nll.item(), 3*math.log(3), places=6)
        nll.backward()
        self.assertTrue(torch.equal(logits.grad[0, 0], torch.zeros(3)))
        self.assertLess(logits.grad[0, 1, 2].item(), 0)
        self.assertTrue(torch.equal(logits.grad[0, 4], torch.zeros(3)))
        with self.assertRaises(ValueError):
            antibody_loss_sum(logits, torch.full((1, 5), -100))

    def test_patience_and_raw_best_checkpoint_are_separate(self):
        monitor = EarlyStopping(patience=2, min_delta=0.01)
        self.assertEqual(monitor.observe(1.0), (True, False))
        self.assertEqual(monitor.observe(0.995), (True, False))
        self.assertEqual(monitor.observe(0.994), (True, True))
        self.assertEqual(monitor.best, 0.994)
        reset = EarlyStopping(patience=2, min_delta=0.01)
        reset.observe(1.0); reset.observe(1.1)
        self.assertEqual(reset.observe(0.98), (True, False))
        self.assertEqual(reset.bad_epochs, 0)

    def test_last_accumulation_window_is_retained(self):
        windows = list(accumulation_windows(list(range(11)), 4))
        self.assertEqual(windows, [[0,1,2,3], [4,5,6,7], [8,9,10]])
        with self.assertRaises(ValueError):
            list(accumulation_windows([1], 0))


if __name__ == '__main__':
    unittest.main()
