import unittest
import warnings

from starlette.exceptions import StarletteDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
    category=StarletteDeprecationWarning,
)
from fastapi.testclient import TestClient

from app import app


class SpringApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_trajectory_endpoint_returns_normalized_rk4_data(self):
        response = self.client.post(
            "/api/trajectory",
            json={"x0": -0.5018395, "v0": 0.9391692, "t_end": 1.0},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["parameters"]["x0"], -0.502)
        self.assertEqual(payload["parameters"]["v0"], 0.939)
        self.assertEqual(payload["parameters"]["t_end"], 1.0)
        self.assertEqual(payload["parameters"]["m"], 1.0)
        self.assertEqual(payload["parameters"]["period"], 2.0)
        self.assertEqual(payload["t"][0], 0.0)
        self.assertEqual(payload["t"][-1], 1.0)
        self.assertEqual(payload["x"][0], -0.502)
        self.assertEqual(payload["v"][0], 0.939)
        self.assertEqual(len(payload["t"]), len(payload["x"]))
        self.assertEqual(len(payload["t"]), len(payload["v"]))

    def test_rejects_out_of_range_inputs(self):
        invalid_payloads = [
            {"x0": -2.001, "v0": 0.0, "t_end": 1.0},
            {"x0": 0.0, "v0": 1.001, "t_end": 1.0},
            {"x0": 0.0, "v0": 0.0, "t_end": 0.09},
            {"x0": 0.0, "v0": 0.0, "t_end": 60.1},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.post("/api/trajectory", json=payload)
                self.assertEqual(response.status_code, 422)

    def test_rejects_missing_extra_and_non_finite_fields(self):
        from math import inf

        from pydantic import ValidationError

        from app import TrajectoryRequest

        missing = self.client.post(
            "/api/trajectory", json={"x0": 0.0, "v0": 0.0}
        )
        extra = self.client.post(
            "/api/trajectory",
            json={"x0": 0.0, "v0": 0.0, "t_end": 1.0, "mass": 3.0},
        )
        self.assertEqual(missing.status_code, 422)
        self.assertEqual(extra.status_code, 422)
        with self.assertRaises(ValidationError):
            TrajectoryRequest(x0=inf, v0=0.0, t_end=1.0)


if __name__ == "__main__":
    unittest.main()
