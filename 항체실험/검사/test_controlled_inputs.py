"""Checks for matched chemistry and explicit topology policy; no docking."""
import copy
import runpy
import tempfile
import tomllib
import unittest
from pathlib import Path

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / '구조평가.py'))


class ControlledInputTests(unittest.TestCase):
    def test_explicit_histidine_policy_is_neutral_and_stable(self):
        block = M['controlled_topology_block']([[100, 35, 212], [60]])
        config = tomllib.loads(block)['topoaa']
        self.assertIs(config['autohis'], False)
        self.assertEqual(config['hydrogen_build'], 'all')
        self.assertEqual(config['mol1']['nhise'], 3)
        self.assertEqual([config['mol1'][f'hise_{i}'] for i in range(1, 4)], [35, 100, 212])
        self.assertEqual(config['mol1']['nhisd'], 0)
        self.assertEqual(config['mol2']['hise_1'], 60)
        self.assertEqual(block, M['controlled_topology_block']([[212, 100, 35], [60]]))

    def test_equal_atom_counts_do_not_hide_charge_or_bond_changes(self):
        data = {'atoms': {'H:1:N': {'charge': -.47, 'type': 'NH1'},
                          'H:1:H': {'charge': .31, 'type': 'H'}}, 'bonds': [['H:1:H', 'H:1:N']]}
        self.assertTrue(M['compare_controlled_chemistry'](data, copy.deepcopy(data))['identical'])
        changed = copy.deepcopy(data); changed['atoms']['H:1:H']['charge'] = .41
        self.assertFalse(M['compare_controlled_chemistry'](data, changed)['identical'])
        changed = copy.deepcopy(data); changed['bonds'] = []
        self.assertFalse(M['compare_controlled_chemistry'](data, changed)['identical'])

    def test_fixed_antigen_comes_from_experiment_not_predicted_complex(self):
        native = M['OUTPUT'] / 'references/CTRL_atezolizumab/resolved_Fv_PDL1.pdb'
        predicted = M['OUTPUT'] / 'control_validation/inputs/CTRL_atezolizumab/predicted_matched_Fv_PDL1.pdb'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'hybrid.pdb'
            M['controlled_partner_structure'](predicted, native, path)
            def ca(source, chain):
                return {line[22:27]: (line[17:20], line[30:54]) for line in source.read_text().splitlines()
                        if line.startswith('ATOM') and line[21] == chain and line[12:16].strip() == 'CA'}
            self.assertEqual(ca(path, 'A'), ca(native, 'A'))
            self.assertEqual(ca(path, 'H'), ca(predicted, 'H'))
            self.assertEqual(ca(path, 'L'), ca(predicted, 'L'))

    def test_cis_type_difference_is_reported_not_hidden_as_identical(self):
        first = {'atoms': {'H:101:C': {'resname': 'TRP', 'charge': .5, 'mass': 12.011, 'type': 'C'}}, 'bonds': []}
        second = copy.deepcopy(first); second['atoms']['H:101:C']['type'] = 'CCIS'
        result = M['compare_controlled_chemistry'](first, second)
        self.assertFalse(result['identical'])
        self.assertTrue(result['protonation_charge_connectivity_identical'])
        self.assertEqual(result['structural_cis_type_differences'], ['H:101:C'])
        second['atoms']['H:101:C']['charge'] = .4
        self.assertFalse(M['compare_controlled_chemistry'](first, second)['protonation_charge_connectivity_identical'])


if __name__ == '__main__':
    unittest.main()
