"""No docking/inference: check selection independence and structural alignment."""
import runpy
import unittest
from pathlib import Path

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / '구조평가.py'))


class AuditTests(unittest.TestCase):
    def test_selection_retains_unclustered_score_leader_without_using_quality(self):
        poses = [dict(model=f'emref_{i}.pdb', score_rank=i, dockq=i / 10) for i in range(1, 9)]
        clusters = ['emref_3.pdb', 'emref_4.pdb', 'emref_5.pdb', 'emref_6.pdb']
        reps = ['emref_3.pdb', 'emref_6.pdb']
        orders = M['control_selection_orders'](poses, clusters, reps)
        self.assertEqual(orders['cluster_first'][:4], clusters)
        self.assertEqual(orders['global_score'][0], 'emref_1.pdb')
        self.assertEqual(orders['diversity_rescue'][0], 'emref_1.pdb')
        for pose in poses:
            pose['dockq'] = 1 - pose['dockq']
        self.assertEqual(orders, M['control_selection_orders'](poses, clusters, reps))
        self.assertEqual(len(orders['diversity_rescue']), len(set(orders['diversity_rescue'])))

    def test_rigid_rotation_translation_are_removed(self):
        import numpy as np
        native = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        predicted = native @ rotation + [9, -3, 5]
        r, t = M['fit_control_coordinates'](predicted, native)
        self.assertLess(np.sqrt(np.mean(np.sum((predicted @ r + t - native)**2, axis=1))), 1e-10)
        self.assertAlmostEqual(np.linalg.det(r), 1.)

    def test_framework_fit_does_not_hide_loop_displacement(self):
        import numpy as np
        framework = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        r, t = M['fit_control_coordinates'](framework + [3, 4, 5], framework)
        moved_loop = np.array([[5., 5, 6]])  # original (1,1,1), +translation, +1A x
        self.assertAlmostEqual(float(np.linalg.norm(moved_loop @ r + t - [[1., 1, 1]])), 1.)


if __name__ == '__main__':
    unittest.main()
