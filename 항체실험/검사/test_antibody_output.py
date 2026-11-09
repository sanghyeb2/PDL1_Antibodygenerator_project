import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from pathlib import Path

from MAGE import parse_antibody
from transformers import PreTrainedTokenizerFast


class OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2]/'MAGE_ab_generation/model_epoch4'
        cls.tokenizer = PreTrainedTokenizerFast.from_pretrained(str(path), local_files_only=True)

    def parse(self, sequence):
        return parse_antibody(self.tokenizer.encode(sequence, add_special_tokens=False), self.tokenizer)

    def test_first_padding_cuts_tail_without_changing_chains(self):
        result = self.parse('A'*110+'[LC]'+'C'*105+'<|pad|>QQQ<|eos|>')
        self.assertEqual(result['vh'], 'A'*110)
        self.assertEqual(result['vl'], 'C'*105)
        self.assertEqual(result['termination_token'], '<|pad|>')
        self.assertTrue(result['basic_sequence_checks_passed'])

    def test_eos_also_marks_sequence_boundary(self):
        self.assertTrue(self.parse('A'*110+'[LC]'+'C'*105+'<|eos|>')['basic_sequence_checks_passed'])

    def test_missing_or_duplicate_chain_boundary_is_not_accepted(self):
        for text in ['A'*220+'<|eos|>', 'A'*110+'[LC]CC[LC]DD<|eos|>']:
            self.assertFalse(self.parse(text)['basic_sequence_checks_passed'])

    def test_length_limited_output_without_termination_is_not_accepted(self):
        self.assertFalse(self.parse('A'*110+'[LC]'+'C'*105)['basic_sequence_checks_passed'])

    def test_short_chain_or_unknown_amino_acid_is_not_accepted(self):
        for text in ['AAA[LC]'+'C'*105+'<|pad|>', 'A'*109+'X[LC]'+'C'*105+'<|pad|>']:
            self.assertFalse(self.parse(text)['basic_sequence_checks_passed'])


if __name__ == '__main__':
    unittest.main()
