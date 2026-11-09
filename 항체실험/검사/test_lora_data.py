import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Regression tests for leakage boundaries and antibody-only supervision."""
import unittest

from LoRA import build_split, encode_example, validate_rows


def row(name, vh='AAAAAAAAAA', vl='CCCCCCCCCC'):
    return dict(id=name, antigen_seq='ACDE', VH_AA=vh, VL_AA=vl,
                SEQ='ACDE[SEP]'+vh+'[LC]'+vl, source_group='fixture')


class SplitTests(unittest.TestCase):
    def test_transitive_or_chain_edges_cannot_cross(self):
        # a~b via VH, b~c via VL, even though a and c differ in both chains.
        rows = [row('a'), row('b', 'AAAAAAAAAT', 'GGGGGGGGGG'),
                row('c', 'TTTTTTTTTT', 'GGGGGGGGGT'),
                row('d', 'YYYYYYYYYY', 'WWWWWWWWWW')]
        result = build_split(rows, threshold=0.9, seed=42, val_fraction=0.25)
        assigned = {r['id']: r for r in result['assignments']}
        self.assertEqual({assigned[k]['group_id'] for k in 'abc'}, {'a'})
        self.assertEqual({assigned[k]['split'] for k in 'abc'}, {'train'})
        self.assertEqual(assigned['d']['split'], 'validation')
        self.assertLess(result['audit']['max_vh_similarity'], 0.9)
        self.assertLess(result['audit']['max_vl_similarity'], 0.9)

    def test_same_chain_and_reordered_input_reproduce(self):
        rows = [row('a'), row('b', 'YYYYYYYYYY'),
                row('c', 'GGGGGGGGGG', 'TTTTTTTTTT'),
                row('d', 'WWWWWWWWWW', 'MMMMMMMMMM')]
        first = build_split(rows, seed=42)
        self.assertEqual(first, build_split(list(reversed(rows)), seed=42))
        assigned = {r['id']: r for r in first['assignments']}
        self.assertEqual(assigned['a']['split'], assigned['b']['split'])

    def test_single_connected_group_is_not_split(self):
        with self.assertRaisesRegex(ValueError, 'connected group'):
            build_split([row('a'), row('b')])

    def test_invalid_inputs_rejected(self):
        bad = row('bad'); bad['SEQ'] = 'incorrect'
        with self.assertRaises(ValueError): validate_rows([bad])
        with self.assertRaises(ValueError): validate_rows([row('a'), row('a')])
        with self.assertRaises(ValueError): validate_rows([row('a', 'AAAAAXAAAA')])


class EncodingTests(unittest.TestCase):
    def test_prompt_and_padding_mask_first_vh_and_eos_supervised(self):
        import os
        from pathlib import Path
        cache = str(Path(__file__).resolve().parents[1]/'캐시/huggingface')
        os.environ.setdefault('HF_HOME', cache)
        os.environ.setdefault('TRANSFORMERS_CACHE', cache+'/hub')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(Path(__file__).resolve().parents[2]/'MAGE_ab_generation/model_epoch4'),
            local_files_only=True)
        r = row('a')
        encoded = encode_example(r, tokenizer, max_length=40)
        sep = encoded['input_ids'].index(tokenizer.convert_tokens_to_ids('[SEP]'))
        self.assertTrue(all(v == -100 for v in encoded['labels'][:sep+1]))
        self.assertEqual(encoded['labels'][sep+1], tokenizer.encode('A')[0])
        eos = encoded['input_ids'].index(tokenizer.eos_token_id)
        self.assertEqual(encoded['labels'][eos], tokenizer.eos_token_id)
        self.assertTrue(all(v == -100 for v in encoded['labels'][eos+1:]))
        self.assertEqual(sum(v != -100 for v in encoded['labels']), 22)
        self.assertEqual(sum(encoded['attention_mask']), 27)
        with self.assertRaisesRegex(ValueError, 'truncat'):
            encode_example(r, tokenizer, max_length=10)


if __name__ == '__main__':
    unittest.main()
