import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import unittest

from LoRA import pair_records, summarize_pairs


class VisualizationDataTests(unittest.TestCase):
    def test_pairs_by_id_not_position_and_weights_by_tokens(self):
        baseline = [dict(id='a', group_id='g', tokens=2, nll=1.),
                    dict(id='b', group_id='h', tokens=6, nll=3.)]
        adapted = [dict(id='b', group_id='h', tokens=6, nll=2.),
                   dict(id='a', group_id='g', tokens=2, nll=0.5)]
        rows = pair_records(baseline, adapted)
        self.assertEqual([r['id'] for r in rows], ['a','b'])
        self.assertEqual([r['delta_nll'] for r in rows], [-0.5,-1.])
        result = summarize_pairs(rows)
        self.assertEqual(result['baseline_nll'], 2.5)
        self.assertEqual(result['lora_nll'], 1.625)
        self.assertEqual(result['improved_pairs'], 2)
        self.assertAlmostEqual(result['lora_perplexity'], math.exp(1.625))

    def test_missing_duplicate_and_changed_targets_fail(self):
        record = dict(id='a',group_id='g',tokens=2,nll=1.)
        with self.assertRaises(ValueError): pair_records([record], [])
        with self.assertRaises(ValueError): pair_records([record, record], [record])
        with self.assertRaises(ValueError): pair_records([record], [{**record,'tokens':3}])
        with self.assertRaises(ValueError): pair_records([record], [{**record,'nll':float('nan')}])


if __name__ == '__main__': unittest.main()
