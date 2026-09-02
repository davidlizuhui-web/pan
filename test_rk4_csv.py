import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class Rk4CsvExportTest(unittest.TestCase):
    def test_script_exports_complete_trajectory_csv_next_to_script(self):
        source_script = Path(__file__).with_name("rk4.py")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            script_dir = temp_root / "script"
            run_dir = temp_root / "run"
            script_dir.mkdir()
            run_dir.mkdir()

            script_path = script_dir / "rk4.py"
            shutil.copy2(source_script, script_path)

            environment = os.environ.copy()
            environment["MPLBACKEND"] = "Agg"
            environment["PYTHONIOENCODING"] = "utf-8"
            subprocess.run(
                [sys.executable, str(script_path)],
                cwd=run_dir,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )

            csv_path = script_dir / "rk4_trajectories.csv"
            self.assertTrue(csv_path.is_file())

            with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(
                list(rows[0]),
                [
                    "trajectory_id",
                    "x0",
                    "v0",
                    "t",
                    "x_clean",
                    "v_clean",
                    "x",
                    "v",
                ],
            )
            self.assertEqual(len(rows), 50_000)
            self.assertEqual(rows[0]["trajectory_id"], "0")
            self.assertEqual(float(rows[0]["t"]), 0.0)
            self.assertEqual(float(rows[0]["x0"]), -0.502)
            self.assertEqual(float(rows[0]["v0"]), 0.939)
            self.assertEqual(float(rows[0]["x_clean"]), -0.502)
            self.assertEqual(float(rows[0]["v_clean"]), 0.939)

            initial_rows = rows[::1000]
            for row in initial_rows:
                self.assertAlmostEqual(float(row["x0"]) * 1000, round(float(row["x0"]) * 1000))
                self.assertAlmostEqual(float(row["v0"]) * 1000, round(float(row["v0"]) * 1000))

            self.assertEqual(rows[-1]["trajectory_id"], "49")
            self.assertEqual(float(rows[-1]["t"]), 10.0)


if __name__ == "__main__":
    unittest.main()
