import csv
from pathlib import Path
import tempfile
import unittest

from rk4_dataset import RK4TrajectoryDataset


class RK4TrajectoryDatasetTest(unittest.TestCase):
    def assert_nested_almost_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_row, expected_row in zip(actual, expected):
            if isinstance(expected_row, list):
                self.assert_nested_almost_equal(actual_row, expected_row)
            else:
                self.assertAlmostEqual(actual_row, expected_row, places=6)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "trajectories.csv"

        rows = [
            [0, -1.0, 0.5, 0.0, -1.0, 0.5, -0.99, 0.51],
            [0, -1.0, 0.5, 0.1, -0.9, 0.6, -0.89, 0.61],
            [0, -1.0, 0.5, 0.2, -0.8, 0.7, -0.79, 0.71],
            [1, 2.0, -0.5, 0.0, 2.0, -0.5, 2.01, -0.49],
            [1, 2.0, -0.5, 0.1, 1.9, -0.6, 1.91, -0.59],
            [1, 2.0, -0.5, 0.2, 1.8, -0.7, 1.81, -0.69],
        ]

        with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "trajectory_id",
                    "x0",
                    "v0",
                    "t",
                    "x_clean",
                    "v_clean",
                    "x",
                    "v",
                ]
            )
            writer.writerows(rows)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_returns_noisy_window_and_next_clean_state(self):
        dataset = RK4TrajectoryDataset(self.csv_path, sequence_length=2)

        inputs, target = dataset[0]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(tuple(inputs.shape), (2, 2))
        self.assertEqual(tuple(target.shape), (2,))
        self.assert_nested_almost_equal(
            inputs.tolist(),
            [[-0.99, 0.51], [-0.89, 0.61]],
        )
        self.assert_nested_almost_equal(target.tolist(), [-0.8, 0.7])

    def test_windows_do_not_cross_trajectory_boundaries(self):
        dataset = RK4TrajectoryDataset(self.csv_path, sequence_length=2)

        inputs, target = dataset[1]

        self.assert_nested_almost_equal(
            inputs.tolist(),
            [[2.01, -0.49], [1.91, -0.59]],
        )
        self.assert_nested_almost_equal(target.tolist(), [1.8, -0.7])

    def test_can_select_trajectories_before_building_windows(self):
        dataset = RK4TrajectoryDataset(
            self.csv_path,
            sequence_length=2,
            trajectory_ids=[1],
        )

        self.assertEqual(len(dataset), 1)
        inputs, _ = dataset[0]
        self.assert_nested_almost_equal(inputs[0].tolist(), [2.01, -0.49])


if __name__ == "__main__":
    unittest.main()
