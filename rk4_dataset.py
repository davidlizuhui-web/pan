import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset


class RK4TrajectoryDataset(Dataset):
    """Create fixed-length state windows from RK4 trajectory CSV data.

    Each item is ``(inputs, target)``. ``inputs`` contains noisy ``x`` and
    ``v`` observations, while ``target`` is the clean state immediately after
    the input window. Windows are built independently for each trajectory.
    """

    REQUIRED_COLUMNS = {
        "trajectory_id",
        "t",
        "x_clean",
        "v_clean",
        "x",
        "v",
    }

    def __init__(self, csv_path="rk4_trajectories.csv", sequence_length=20, trajectory_ids=None):
        if not isinstance(sequence_length, int) or sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer")

        self.csv_path = Path(csv_path)
        self.sequence_length = sequence_length
        selected_ids = None if trajectory_ids is None else set(trajectory_ids)

        rows_by_trajectory = defaultdict(list)
        with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            columns = set(reader.fieldnames or [])
            missing_columns = self.REQUIRED_COLUMNS - columns
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"CSV is missing required columns: {missing}")

            for row in reader:
                trajectory_id = int(row["trajectory_id"])
                if selected_ids is not None and trajectory_id not in selected_ids:
                    continue

                rows_by_trajectory[trajectory_id].append(
                    (
                        float(row["t"]),
                        float(row["x"]),
                        float(row["v"]),
                        float(row["x_clean"]),
                        float(row["v_clean"]),
                    )
                )

        self._trajectories = {}
        self._sample_index = []

        for trajectory_id in sorted(rows_by_trajectory):
            rows = sorted(rows_by_trajectory[trajectory_id], key=lambda row: row[0])
            inputs = torch.tensor(
                [[row[1], row[2]] for row in rows],
                dtype=torch.float32,
            )
            clean_states = torch.tensor(
                [[row[3], row[4]] for row in rows],
                dtype=torch.float32,
            )
            self._trajectories[trajectory_id] = (inputs, clean_states)

            window_count = max(0, len(rows) - self.sequence_length)
            self._sample_index.extend(
                (trajectory_id, start) for start in range(window_count)
            )

    def __len__(self):
        return len(self._sample_index)

    def __getitem__(self, index):
        trajectory_id, start = self._sample_index[index]
        inputs, clean_states = self._trajectories[trajectory_id]
        stop = start + self.sequence_length

        return inputs[start:stop], clean_states[stop]

if __name__ == "__main__":
    data = RK4TrajectoryDataset()