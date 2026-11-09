import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import json
from pathlib import Path
import tempfile
import unittest

import MAGE
import LoRA


class SingleFileTests(unittest.TestCase):
    def test_default_modes_and_paths(self):
        self.assertEqual(MAGE.build_parser().parse_args([]).mode, 'generate')
        self.assertEqual(LoRA.build_parser().parse_args([]).mode, 'generate')
        self.assertTrue(MAGE.CHECKPOINT.is_dir())
        self.assertTrue(LoRA.DEFAULT_ADAPTER.is_dir())

    def test_antigen_is_explicit_or_unique_csv_antigen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.csv'
            path.write_text('antigen_seq\nACDE\nACDE\n',encoding='utf-8')
            self.assertEqual(MAGE.read_antigen(path), 'ACDE')
            path.write_text('antigen_seq\nACDE\nACDF\n',encoding='utf-8')
            with self.assertRaises(ValueError): MAGE.read_antigen(path)

    def test_both_files_encode_same_21_targets_and_keep_split(self):
        manifest, datasets=LoRA.load_verified_split(LoRA.DEFAULT_DATA, LoRA.DEFAULT_SPLIT)
        self.assertEqual(manifest['counts']['train'],85)
        self.assertEqual(len(datasets['validation']),21)
        from transformers import PreTrainedTokenizerFast
        tokenizer=PreTrainedTokenizerFast.from_pretrained(str(MAGE.CHECKPOINT),local_files_only=True)
        for row in datasets['validation']:
            self.assertEqual(MAGE.encode_example(row,tokenizer),LoRA.encode_example(row,tokenizer))
        with tempfile.TemporaryDirectory() as tmp:
            split=Path(tmp)
            for name in ('split_manifest.json','train.csv','validation.csv'):
                (split/name).write_bytes((LoRA.DEFAULT_SPLIT/name).read_bytes())
            rows=datasets['validation']
            rows[0]['VH_AA']='A'+rows[0]['VH_AA'][1:]
            with (split/'validation.csv').open('w',encoding='utf-8-sig',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with self.assertRaises(ValueError): LoRA.load_verified_split(LoRA.DEFAULT_DATA,split)
            with self.assertRaises(ValueError): MAGE.verified_validation_rows(MAGE.DEFAULT_DATA,split)


if __name__=='__main__': unittest.main()
