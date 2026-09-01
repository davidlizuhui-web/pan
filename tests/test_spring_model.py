import math
import unittest

import numpy as np

from spring_model import solve_trajectory


class SolveTrajectoryTest(unittest.TestCase):
    def test_rounds_initial_values_and_preserves_initial_state(self):
        trajectory = solve_trajectory(-0.5018395, 0.9391692, 1.0)

        self.assertEqual(trajectory.x0, -0.502)
        self.assertEqual(trajectory.v0, 0.939)
        self.assertEqual(trajectory.x[0], -0.502)
        self.assertEqual(trajectory.v[0], 0.939)

    def test_includes_exact_endpoint_with_equal_array_lengths(self):
        trajectory = solve_trajectory(1.0, 0.0, 1.005)

        self.assertEqual(trajectory.t[0], 0.0)
        self.assertEqual(trajectory.t[-1], 1.005)
        self.assertEqual(len(trajectory.t), len(trajectory.x))
        self.assertEqual(len(trajectory.t), len(trajectory.v))
        self.assertLessEqual(float(np.max(np.diff(trajectory.t))), 0.01)

    def test_matches_analytical_solution(self):
        trajectory = solve_trajectory(1.0, 0.25, 10.0)
        expected_x = math.cos(math.pi * 10.0) + (0.25 / math.pi) * math.sin(
            math.pi * 10.0
        )
        expected_v = -math.pi * math.sin(math.pi * 10.0) + 0.25 * math.cos(
            math.pi * 10.0
        )

        self.assertLess(abs(trajectory.x[-1] - expected_x), 1e-6)
        self.assertLess(abs(trajectory.v[-1] - expected_v), 1e-6)

    def test_rejects_non_finite_or_non_positive_solver_inputs(self):
        invalid_arguments = [
            (math.nan, 0.0, 1.0, 0.01),
            (0.0, math.inf, 1.0, 0.01),
            (0.0, 0.0, 0.0, 0.01),
            (0.0, 0.0, 1.0, 0.0),
        ]
        for x0, v0, t_end, dt in invalid_arguments:
            with self.subTest(x0=x0, v0=v0, t_end=t_end, dt=dt):
                with self.assertRaises(ValueError):
                    solve_trajectory(x0, v0, t_end, dt)


if __name__ == "__main__":
    unittest.main()
