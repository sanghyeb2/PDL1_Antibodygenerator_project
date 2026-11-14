import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest

from LoRA import rejection_reasons, unique_selection, enable_generation_cache, select_paper_candidates, add_reference_metrics, REGIONS, acquire_generation_lock


class CampaignTests(unittest.TestCase):
    def test_generation_lock_prevents_second_writer_and_releases(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            first=acquire_generation_lock(Path(folder))
            self.assertIsNotNone(first)
            self.assertIsNone(acquire_generation_lock(Path(folder)))
            first.close()
            resumed=acquire_generation_lock(Path(folder))
            self.assertIsNotNone(resumed);resumed.close()

    def test_region_plot_uses_same_nearest_full_chain_reference(self):
        regions={f'{r}.{c}':'AA' for c in ('H','L') for r in REGIONS}
        query=dict(VH_AA='AAAA',VL_AA='CCCC',**regions)
        first=dict(id='first',VH_AA='AAAA',VL_AA='CCCC',**regions);first['CDR3.H']='GG'
        second=dict(id='second',VH_AA='ZZZZ',VL_AA='TTTT',**regions)
        add_reference_metrics([query],{'test':[first,second]})
        self.assertEqual(query['min_edit_test_CDR3.H'],0)
        self.assertEqual(query['chain_match_edit_test_CDR3.H'],2)
        self.assertEqual(query['nearest_chain_reference_test_VH_AA'],'first')

    def test_selection_preserves_gates_and_reports_shortages(self):
        def row(i,**extra):
            return dict(id=str(i),VH_AA=str(i),VL_AA='L',humanness_pass=True,
                        **{'v_identity.H':.9,'v_gene.H':'IGHV1-2',
                           'max_identity_original_CDR3.H':.8,'max_identity_train85_CDR3.H':.7,
                           'max_identity_known106_CDR3.H':.8,'max_identity_known106_VH_AA':.8},**extra)
        good=row(1)
        bad=row(2);bad['max_identity_train85_CDR3.H']=1
        germline=row(3);germline['v_identity.H']=1
        chosen,report=select_paper_candidates([good,bad,germline])
        self.assertEqual([r['id'] for r in chosen],['1'])
        self.assertEqual(report['eligible_unique_pairs'],1)
        self.assertEqual(report['shortage'],19)

    def test_failure_reasons_preserve_all_invalid_attempts(self):
        parsed=dict(termination_found=False,lc_token_count=0,standard_amino_acids_only=False,
                    vh_length=None,vl_length=None)
        self.assertIn('missing_termination',rejection_reasons(parsed))
        self.assertIn('invalid_lc_count',rejection_reasons(parsed))
        good=dict(termination_found=True,lc_token_count=1,standard_amino_acids_only=True,
                  vh_length=100,vl_length=105)
        self.assertEqual(rejection_reasons(good),[])
        self.assertIn('short_vh',rejection_reasons({**good,'vh_length':99}))

    def test_final_selection_never_repeats_identical_pairs(self):
        rows=[dict(id='a',VH_AA='AA',VL_AA='CC'),dict(id='b',VH_AA='AA',VL_AA='CC'),
              dict(id='c',VH_AA='GG',VL_AA='TT')]
        self.assertEqual([r['id'] for r in unique_selection(rows,3)],['a','c'])

    def test_progen_cache_is_used_and_matches_full_forward(self):
        import torch
        import MAGE
        sys.path.insert(0,str(MAGE.SOURCE))
        from progen.modeling_progen import ProGenForCausalLM
        from progen.configuration_progen import ProGenConfig
        torch.manual_seed(7)
        model=ProGenForCausalLM(ProGenConfig(vocab_size=32,n_positions=64,n_ctx=64,
                     n_embd=64,n_layer=1,n_head=8,rotary_dim=8)).eval()
        enable_generation_cache(model)
        ids=torch.tensor([[1,3,4]])
        with torch.inference_mode():
            cache=model(ids,use_cache=True).past_key_values
            prepared=model.prepare_inputs_for_generation(torch.tensor([[1,3,4,5]]),
                           past_key_values=cache,attention_mask=torch.ones(1,4,dtype=torch.long),use_cache=True)
            self.assertEqual(prepared['input_ids'].shape[1],1)
            cached=model(**prepared).logits[:,-1]
            full=model(torch.tensor([[1,3,4,5]]),use_cache=False).logits[:,-1]
        torch.testing.assert_close(cached,full,rtol=1e-4,atol=1e-5)

    def test_peft_generate_keeps_cached_single_token_decoding(self):
        import torch
        import MAGE
        from peft import LoraConfig, get_peft_model
        sys.path.insert(0,str(MAGE.SOURCE))
        from progen.modeling_progen import ProGenForCausalLM
        from progen.configuration_progen import ProGenConfig
        base=ProGenForCausalLM(ProGenConfig(vocab_size=32,n_positions=64,n_ctx=64,
                     n_embd=64,n_layer=1,n_head=8,rotary_dim=8)).eval()
        model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['qkv_proj','out_proj'],task_type='CAUSAL_LM')).eval()
        enable_generation_cache(model)
        lengths=[]
        handle=base.register_forward_pre_hook(lambda module,args,kwargs:lengths.append(kwargs['input_ids'].shape[1]),with_kwargs=True)
        with torch.inference_mode():
            model.generate(input_ids=torch.tensor([[1,3,4]]),max_new_tokens=3,
                           do_sample=False,pad_token_id=0,eos_token_id=[],use_cache=True)
        handle.remove()
        self.assertEqual(lengths,[3,1,1])


if __name__=='__main__': unittest.main()
