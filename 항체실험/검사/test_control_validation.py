"""Scientific sanity checks; no docking or model inference is repeated."""
import runpy
import tempfile
import unittest
from pathlib import Path

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / '구조평가.py'))


class ControlValidationTests(unittest.TestCase):
    def test_selected_cluster_names_map_back_to_emref(self):
        root = MODULE['ENV'] / 'runs/main/CTRL_atezolizumab'
        names = MODULE['selected_control_models'](root)
        self.assertEqual(len(names), 4)
        self.assertEqual(names[0], 'emref_33.pdb')
        self.assertTrue(all((root / 'run/4_emref' / (name + '.gz')).exists()
                            or (root / 'run/4_emref' / name).exists() for name in names))

    def test_native_self_and_displaced_antibody(self):
        reference = MODULE['OUTPUT'] / 'references/CTRL_atezolizumab/resolved_Fv_PDL1.pdb'
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            same = MODULE['capri_quality'](reference, reference, directory / 'self')
            self.assertAlmostEqual(same['dockq'], 1., places=5)
            self.assertAlmostEqual(same['fnat'], 1., places=5)
            moved = directory / 'moved.pdb'
            lines = []
            for line in reference.read_text().splitlines():
                if line.startswith('ATOM') and line[21] in 'HL':
                    line = line[:30] + f'{float(line[30:38]) + 100:8.3f}' + line[38:]
                lines.append(line)
            moved.write_text('\n'.join(lines) + '\n')
            bad = MODULE['capri_quality'](moved, reference, directory / 'moved')
            self.assertEqual(bad['fnat'], 0.)
            self.assertLess(bad['dockq'], .23)

    def test_matched_prediction_preserves_predicted_coordinates(self):
        row = next(r for r in MODULE['read_json'](MODULE['OUTPUT'] / 'manifest.json')['candidates']
                   if r['id'] == 'CTRL_atezolizumab')
        with tempfile.TemporaryDirectory() as directory:
            result = MODULE['matched_validation_structure'](row, Path(directory))
            def atoms(path):
                return {(l[21], l[22:27], l[12:16]): l[30:54]
                        for l in path.read_text().splitlines() if l.startswith('ATOM')}
            result_atoms = atoms(result)
            predicted = atoms(Path(directory) / 'predicted_Fv.pdb') | atoms(Path(directory) / 'predicted_antigen.pdb')
            self.assertTrue(result_atoms)
            self.assertTrue(all(predicted[k] == v for k, v in result_atoms.items()))
            reference = MODULE['OUTPUT'] / 'references' / row['id'] / 'resolved_Fv_PDL1.pdb'
            self.assertEqual({k[:2] for k in result_atoms}, {k[:2] for k in atoms(reference)})


if __name__ == '__main__':
    unittest.main()
