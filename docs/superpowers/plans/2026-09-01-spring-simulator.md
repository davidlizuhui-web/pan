# Interactive Mass-Spring Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI-served browser application that computes an ideal mass-spring trajectory with Python RK4 and lets the user play, pause, reset, and seek through a synchronized spring animation and displacement plot.

**Architecture:** A focused `spring_model.py` module performs all numerical integration. FastAPI validates inputs and returns a complete trajectory as JSON; native JavaScript keeps that trajectory in browser memory and uses `requestAnimationFrame` for smooth local playback without per-frame server requests. SVG renders both the spring/mass scene and the displacement plot.

**Tech Stack:** Python 3.11.15, NumPy 2.4.6, FastAPI 0.139.0, Pydantic 2.13.4, Uvicorn 0.50.2, httpx 0.28.1, standard-library `unittest`, HTML5, CSS, native JavaScript, SVG.

**Spec:** `docs/superpowers/specs/2026-09-01-spring-simulator-design.md`

## Global Constraints

- Run every Python command with `D:\miniconda\envs\resnet\python.exe` or from an activated `resnet` Conda environment.
- Keep the physical parameters fixed at `m=1.0`, `k=pi^2`, `omega=pi`, with no damping, external force, or measurement noise.
- Round `x0` and `v0` to exactly three decimal places before RK4 integration.
- Accept `x0` only in `[-2.0, 2.0]`, `v0` only in `[-1.0, 1.0]`, and `t_end` only in `[0.1, 60.0]`.
- Use a target integration step of `0.01` seconds and include the exact `t_end` endpoint.
- Keep playback entirely in the browser after one trajectory response; do not poll the server per animation frame.
- Use no npm packages or JavaScript frameworks; Node and npm are not installed.
- Do not alter or regenerate `gen.py`, `rk4.py`, `rk4_trajectories.csv`, or `test_rk4_csv.py` as part of this feature.
- Preserve unrelated user changes in the dirty working tree and stage only task-specific files for each commit.

## File Structure

- Create `spring_model.py`: physical constants, trajectory value object, derivative function, RK4 step, and complete solver.
- Create `tests/__init__.py`: package marker so focused `python -m unittest tests.test_*` commands resolve consistently.
- Create `app.py`: Pydantic API models, health endpoint, trajectory endpoint, index route, and static mounting.
- Create `static/index.html`: accessible control form, SVG scenes, current-value readouts, and playback controls.
- Create `static/style.css`: responsive visual system and all control/simulation styling.
- Create `static/app.js`: API client, state machine, interpolation, playback clock, spring renderer, and plot renderer.
- Create `tests/test_spring_model.py`: numerical and boundary tests for the model module.
- Create `tests/test_api.py`: request validation and response-contract tests.
- Create `tests/test_frontend.py`: index/static-resource and HTML-control contract tests.
- Create `tests/test_browser_smoke.py`: real headless-Chrome verification that the default API response is rendered into SVG and live readouts.

---

### Task 1: RK4 Physics Core

**Files:**
- Create: `spring_model.py`
- Create: `tests/__init__.py`
- Create: `tests/test_spring_model.py`

**Interfaces:**
- Consumes: scalar `x0: float`, `v0: float`, `t_end: float`, and optional `dt: float = 0.01`.
- Produces: constants `MASS`, `SPRING_CONSTANT`, `OMEGA`, `PERIOD`; `Trajectory` dataclass; `derivatives(x, v)`; `rk4_step(x, v, dt)`; `solve_trajectory(x0, v0, t_end, dt=0.01)`.
- `Trajectory` fields are `x0: float`, `v0: float`, `t_end: float`, `t: numpy.ndarray`, `x: numpy.ndarray`, and `v: numpy.ndarray`.

- [ ] **Step 1: Create the test package and write failing tests for normalization, endpoints, and array shape**

Create an empty `tests/__init__.py`, then create `tests/test_spring_model.py` with the initial contract tests:

```python
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
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run:

```powershell
& 'D:\miniconda\envs\resnet\python.exe' -m unittest -v tests.test_spring_model
```

Expected: error with `ModuleNotFoundError: No module named 'spring_model'`.

- [ ] **Step 3: Implement the minimal trajectory solver**

Create `spring_model.py` with these definitions:

```python
from dataclasses import dataclass
import math

import numpy as np


MASS = 1.0
SPRING_CONSTANT = math.pi**2
OMEGA = math.sqrt(SPRING_CONSTANT / MASS)
PERIOD = 2.0 * math.pi / OMEGA


@dataclass(frozen=True)
class Trajectory:
    x0: float
    v0: float
    t_end: float
    t: np.ndarray
    x: np.ndarray
    v: np.ndarray


def derivatives(x: float, v: float) -> tuple[float, float]:
    return v, -(OMEGA**2) * x


def rk4_step(x: float, v: float, dt: float) -> tuple[float, float]:
    k1_x, k1_v = derivatives(x, v)
    k2_x, k2_v = derivatives(x + 0.5 * dt * k1_x, v + 0.5 * dt * k1_v)
    k3_x, k3_v = derivatives(x + 0.5 * dt * k2_x, v + 0.5 * dt * k2_v)
    k4_x, k4_v = derivatives(x + dt * k3_x, v + dt * k3_v)
    return (
        x + dt * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x) / 6.0,
        v + dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0,
    )


def solve_trajectory(
    x0: float,
    v0: float,
    t_end: float,
    dt: float = 0.01,
) -> Trajectory:
    normalized_x0 = round(float(x0), 3)
    normalized_v0 = round(float(v0), 3)
    step_count = max(1, math.ceil(float(t_end) / float(dt)))
    times = np.linspace(0.0, float(t_end), step_count + 1)
    positions = np.empty(step_count + 1, dtype=float)
    velocities = np.empty(step_count + 1, dtype=float)
    positions[0] = normalized_x0
    velocities[0] = normalized_v0

    for index in range(step_count):
        actual_dt = float(times[index + 1] - times[index])
        positions[index + 1], velocities[index + 1] = rk4_step(
            positions[index], velocities[index], actual_dt
        )

    return Trajectory(
        x0=normalized_x0,
        v0=normalized_v0,
        t_end=float(t_end),
        t=times,
        x=positions,
        v=velocities,
    )
```

- [ ] **Step 4: Run the focused tests and verify they pass**

Run the same unittest command. Expected: two tests pass.

- [ ] **Step 5: Add failing analytical-accuracy and invalid-input tests**

Append to `SolveTrajectoryTest`:

```python
    def test_matches_analytical_solution(self):
        trajectory = solve_trajectory(1.0, 0.25, 10.0)
        expected_x = math.cos(math.pi * 10.0) + (0.25 / math.pi) * math.sin(math.pi * 10.0)
        expected_v = -math.pi * math.sin(math.pi * 10.0) + 0.25 * math.cos(math.pi * 10.0)

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
```

- [ ] **Step 6: Run the focused tests and verify the validation test fails**

Expected: the analytical test passes or is within tolerance; invalid inputs fail because validation is absent.

- [ ] **Step 7: Add finite and positivity validation before normalization**

Insert at the start of `solve_trajectory`:

```python
    values = (float(x0), float(v0), float(t_end), float(dt))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("x0, v0, t_end, and dt must be finite")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
```

Use the validated float values for subsequent computation so the function does not convert the same argument multiple times.

- [ ] **Step 8: Run the full physics test module**

Expected: four tests pass with no warnings.

- [ ] **Step 9: Commit the physics core**

```powershell
git add -- spring_model.py tests/__init__.py tests/test_spring_model.py
git commit -m "feat: add reusable RK4 spring solver"
```

---

### Task 2: FastAPI Trajectory API

**Files:**
- Create: `app.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes: `spring_model.solve_trajectory`, physical constants, and JSON `{x0, v0, t_end}`.
- Produces: `GET /health` and `POST /api/trajectory`.
- `POST /api/trajectory` returns `parameters`, `t`, `x`, and `v`; all arrays are JSON number lists with equal lengths.

- [ ] **Step 1: Write failing health and trajectory API tests**

Create `tests/test_api.py`:

```python
import unittest

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
```

- [ ] **Step 2: Run the API tests and verify the import fails**

Run:

```powershell
& 'D:\miniconda\envs\resnet\python.exe' -m unittest -v tests.test_api
```

Expected: error with `ModuleNotFoundError: No module named 'app'`.

- [ ] **Step 3: Implement Pydantic models and the two endpoints**

Create `app.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel

from spring_model import MASS, OMEGA, PERIOD, SPRING_CONSTANT, solve_trajectory


class TrajectoryRequest(BaseModel):
    x0: float
    v0: float
    t_end: float


class ParametersResponse(BaseModel):
    x0: float
    v0: float
    t_end: float
    m: float
    k: float
    omega: float
    period: float


class TrajectoryResponse(BaseModel):
    parameters: ParametersResponse
    t: list[float]
    x: list[float]
    v: list[float]


app = FastAPI(title="Interactive Mass-Spring Simulator")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/trajectory", response_model=TrajectoryResponse)
def create_trajectory(request: TrajectoryRequest) -> TrajectoryResponse:
    trajectory = solve_trajectory(request.x0, request.v0, request.t_end)
    return TrajectoryResponse(
        parameters=ParametersResponse(
            x0=trajectory.x0,
            v0=trajectory.v0,
            t_end=trajectory.t_end,
            m=MASS,
            k=SPRING_CONSTANT,
            omega=OMEGA,
            period=PERIOD,
        ),
        t=trajectory.t.tolist(),
        x=trajectory.x.tolist(),
        v=trajectory.v.tolist(),
    )
```

- [ ] **Step 4: Run the focused API tests**

Expected: both tests pass.

- [ ] **Step 5: Add failing boundary and malformed-request tests**

Append to `SpringApiTest`:

```python
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

        missing = self.client.post("/api/trajectory", json={"x0": 0.0, "v0": 0.0})
        extra = self.client.post(
            "/api/trajectory",
            json={"x0": 0.0, "v0": 0.0, "t_end": 1.0, "mass": 3.0},
        )
        self.assertEqual(missing.status_code, 422)
        self.assertEqual(extra.status_code, 422)
        with self.assertRaises(ValidationError):
            TrajectoryRequest(x0=inf, v0=0.0, t_end=1.0)
```

Run the tests now. Expected: out-of-range requests return `200`, the extra field is ignored, and infinity is accepted by the request model, so the new tests fail for the intended missing-validation behavior.

- [ ] **Step 6: Add strict Pydantic validation and rerun all API tests**

Replace `TrajectoryRequest` and its imports with:

```python
from pydantic import BaseModel, ConfigDict, Field


class TrajectoryRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    x0: float = Field(ge=-2.0, le=2.0)
    v0: float = Field(ge=-1.0, le=1.0)
    t_end: float = Field(ge=0.1, le=60.0)
```

Expected: four tests pass.

- [ ] **Step 7: Run all Python tests accumulated so far**

```powershell
& 'D:\miniconda\envs\resnet\python.exe' -m unittest discover -s tests -v
```

Expected: eight tests pass.

- [ ] **Step 8: Commit the API**

```powershell
git add -- app.py tests/test_api.py
git commit -m "feat: expose RK4 trajectory API"
```

---

### Task 3: Interactive Browser Interface

**Files:**
- Modify: `app.py`
- Create: `static/index.html`
- Create: `static/style.css`
- Create: `static/app.js`
- Create: `tests/test_frontend.py`
- Create: `tests/test_browser_smoke.py`

**Interfaces:**
- Consumes: `POST /api/trajectory` response from Task 2.
- Produces: `GET /`, `/static/style.css`, `/static/app.js`, the control element IDs listed below, and a browser playback state with `trajectory`, `currentTime`, `playing`, `speed`, `lastFrameTime`, and `animationFrameId`.
- Required HTML IDs: `parameter-form`, `x0-input`, `v0-input`, `duration-input`, `calculate-button`, `play-button`, `pause-button`, `reset-button`, `speed-select`, `time-slider`, `time-value`, `position-value`, `velocity-value`, `error-message`, `simulation-svg`, `spring-path`, `mass-block`, `plot-svg`, `trajectory-path`, `time-guide`, and `current-point`.

- [ ] **Step 1: Write failing static-resource and HTML-contract tests**

Create `tests/test_frontend.py`:

```python
from html.parser import HTMLParser
import unittest

from fastapi.testclient import TestClient

from app import app


class IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])


class FrontendContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_index_exposes_all_interactive_controls(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

        parser = IdCollector()
        parser.feed(response.text)
        required_ids = {
            "parameter-form", "x0-input", "v0-input", "duration-input",
            "calculate-button", "play-button", "pause-button", "reset-button",
            "speed-select", "time-slider", "time-value", "position-value",
            "velocity-value", "error-message", "simulation-svg", "spring-path",
            "mass-block", "plot-svg", "trajectory-path", "time-guide",
            "current-point",
        }
        self.assertEqual(required_ids - parser.ids, set())

    def test_frontend_assets_are_served(self):
        style = self.client.get("/static/style.css")
        script = self.client.get("/static/app.js")
        self.assertEqual(style.status_code, 200)
        self.assertIn("text/css", style.headers["content-type"])
        self.assertEqual(script.status_code, 200)
        self.assertIn("javascript", script.headers["content-type"])
```

Also create `tests/test_browser_smoke.py` so the JavaScript behavior is tested through a real browser rather than by matching source text:

```python
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


class BrowserSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            cls.port = probe.getsockname()[1]
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        cls.server = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "app:app", "--host",
                "127.0.0.1", "--port", str(cls.port),
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        health_url = f"http://127.0.0.1:{cls.port}/health"
        for _ in range(50):
            try:
                with urlopen(health_url, timeout=0.2) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(0.1)
        cls.server.terminate()
        raise RuntimeError("Uvicorn did not become ready")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        try:
            cls.server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.server.kill()
            cls.server.wait(timeout=5)

    @unittest.skipUnless(CHROME.is_file(), "Chrome is required for browser smoke testing")
    def test_default_trajectory_is_rendered_by_javascript(self):
        with tempfile.TemporaryDirectory() as profile:
            result = subprocess.run(
                [
                    str(CHROME), "--headless=new", "--disable-gpu",
                    "--virtual-time-budget=3000", f"--user-data-dir={profile}",
                    "--dump-dom", f"http://127.0.0.1:{self.port}/",
                ],
                check=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )

        html = result.stdout
        self.assertRegex(html, r'id="position-value">-0\.502<')
        self.assertRegex(html, r'id="velocity-value">0\.939<')
        self.assertRegex(html, r'id="spring-path"[^>]*d="M[^\"]+"')
        self.assertRegex(html, r'id="trajectory-path"[^>]*d="M[^\"]+"')
```

- [ ] **Step 2: Run the frontend tests and verify the root route fails**

Run:

```powershell
& 'D:\miniconda\envs\resnet\python.exe' -m unittest -v tests.test_frontend
```

Also run `tests.test_browser_smoke`. Expected: `GET /` returns `404`, the HTML contract test fails, and the browser smoke test cannot find rendered default values.

- [ ] **Step 3: Create the complete semantic page shell**

Create `static/index.html` with this structure and the exact IDs from the contract:

```html
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>交互式弹簧振子</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <main class="app-shell">
    <header class="hero">
      <p class="eyebrow">RK4 · MASS–SPRING LAB</p>
      <h1>交互式弹簧振子</h1>
      <p>输入初始位移和速度，观察状态如何随时间演化。</p>
    </header>
    <section class="workspace">
      <aside class="control-panel" aria-label="模拟参数">
        <form id="parameter-form">
          <label>初始位移 x₀<input id="x0-input" type="number" min="-2" max="2" step="0.001" value="-0.502" required></label>
          <label>初始速度 v₀<input id="v0-input" type="number" min="-1" max="1" step="0.001" value="0.939" required></label>
          <label>总时长 t<input id="duration-input" type="number" min="0.1" max="60" step="0.1" value="10" required></label>
          <button id="calculate-button" type="submit">计算轨迹</button>
        </form>
        <div id="error-message" role="alert" aria-live="polite"></div>
        <div class="transport" aria-label="播放控制">
          <button id="play-button" type="button">播放</button>
          <button id="pause-button" type="button">暂停</button>
          <button id="reset-button" type="button">重置</button>
          <label>速度<select id="speed-select"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label>
        </div>
      </aside>
      <div class="visual-panel">
        <div class="readouts">
          <output>t <strong id="time-value">0.000</strong> s</output>
          <output>x <strong id="position-value">—</strong></output>
          <output>v <strong id="velocity-value">—</strong></output>
        </div>
        <svg id="simulation-svg" viewBox="0 0 900 320" role="img" aria-label="弹簧与物体运动示意图">
          <path class="wall" d="M80 55V270"></path>
          <path id="spring-path" class="spring"></path>
          <rect id="mass-block" class="mass" width="112" height="96" rx="16"></rect>
        </svg>
        <label class="timeline">时间<input id="time-slider" type="range" min="0" max="10" step="0.001" value="0"></label>
        <svg id="plot-svg" viewBox="0 0 900 300" role="img" aria-label="位移时间曲线">
          <path id="trajectory-path" class="trajectory"></path>
          <line id="time-guide" class="time-guide"></line>
          <circle id="current-point" class="current-point" r="7"></circle>
        </svg>
      </div>
    </section>
  </main>
  <script src="/static/app.js" defer></script>
</body>
</html>
```

Add SVG guide lines, equilibrium marker, axis labels, and accessible button labels without changing the required IDs.

- [ ] **Step 4: Implement responsive styling**

Create `static/style.css`. Define a restrained laboratory visual system with CSS variables and these concrete behaviors:

```css
:root {
  color-scheme: dark;
  --bg: #07111f;
  --panel: rgba(14, 29, 49, 0.88);
  --panel-strong: #11243b;
  --line: rgba(166, 205, 230, 0.18);
  --text: #eef7fb;
  --muted: #9db4c5;
  --cyan: #55d6d0;
  --amber: #ffb454;
  --danger: #ff7d86;
  font-family: Inter, "Segoe UI", sans-serif;
}

* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; background: radial-gradient(circle at 75% 10%, #143b54 0, var(--bg) 46%); color: var(--text); }
.app-shell { width: min(1400px, calc(100% - 32px)); margin: 0 auto; padding: 40px 0 56px; }
.workspace { display: grid; grid-template-columns: minmax(260px, 340px) minmax(0, 1fr); gap: 24px; }
.control-panel, .visual-panel { border: 1px solid var(--line); border-radius: 24px; background: var(--panel); box-shadow: 0 24px 80px rgba(0, 0, 0, .24); }
.control-panel { padding: 24px; }
.visual-panel { padding: 20px; overflow: hidden; }
.control-panel form, .transport { display: grid; gap: 14px; }
label { display: grid; gap: 8px; color: var(--muted); }
input, select, button { min-height: 44px; border-radius: 12px; border: 1px solid var(--line); font: inherit; }
button { cursor: pointer; background: var(--panel-strong); color: var(--text); }
#calculate-button { background: var(--cyan); color: #031417; font-weight: 700; }
.readouts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.spring { fill: none; stroke: var(--cyan); stroke-width: 7; stroke-linecap: round; stroke-linejoin: round; }
.mass { fill: var(--amber); filter: drop-shadow(0 14px 18px rgba(255, 180, 84, .22)); }
.trajectory { fill: none; stroke: var(--cyan); stroke-width: 4; }
.time-guide { stroke: rgba(255, 255, 255, .35); stroke-dasharray: 6 6; }
.current-point { fill: var(--amber); }
#error-message:not(:empty) { margin-top: 14px; color: var(--danger); }
button:disabled, input:disabled, select:disabled { cursor: not-allowed; opacity: .5; }

@media (max-width: 850px) {
  .workspace { grid-template-columns: 1fr; }
  .readouts { grid-template-columns: 1fr; }
}
```

Complete spacing, focus-visible states, plot axes, slider, hover states, and the wall/guide styling in the same visual language. Do not introduce external fonts or image assets.

- [ ] **Step 5: Implement the frontend state machine and API request**

Create `static/app.js` with strict mode and one DOM lookup map. Use this state shape:

```javascript
"use strict";

const state = {
  trajectory: null,
  currentTime: 0,
  playing: false,
  speed: 1,
  loading: false,
  lastFrameTime: null,
  animationFrameId: null,
};

const elements = Object.fromEntries([
  "parameter-form", "x0-input", "v0-input", "duration-input",
  "calculate-button", "play-button", "pause-button", "reset-button",
  "speed-select", "time-slider", "time-value", "position-value",
  "velocity-value", "error-message", "spring-path", "mass-block",
  "trajectory-path", "time-guide", "current-point",
].map((id) => [id, document.getElementById(id)]));
```

Implement `requestTrajectory()` to call the API only when the form is submitted or on initial page load:

```javascript
async function requestTrajectory() {
  pausePlayback();
  setLoading(true);
  showError("");
  const payload = {
    x0: Number(elements["x0-input"].value),
    v0: Number(elements["v0-input"].value),
    t_end: Number(elements["duration-input"].value),
  };

  try {
    const response = await fetch("/api/trajectory", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(validationMessage(body));
    validateTrajectory(body);
    state.trajectory = body;
    state.currentTime = 0;
    elements["x0-input"].value = body.parameters.x0.toFixed(3);
    elements["v0-input"].value = body.parameters.v0.toFixed(3);
    elements["time-slider"].max = String(body.parameters.t_end);
    elements["time-slider"].value = "0";
    renderTrajectoryPath();
    renderCurrentState();
  } catch (error) {
    showError(error instanceof Error ? error.message : "无法计算轨迹");
  } finally {
    setLoading(false);
    updateControlState();
  }
}
```

`validateTrajectory(body)` must reject missing/non-array `t`, `x`, or `v`, unequal or empty arrays, non-finite values, and an endpoint that differs from `parameters.t_end`. `validationMessage(body)` should return the first FastAPI validation message when `body.detail` is an array and otherwise return `"请求失败，请检查输入参数"`.

Use these concrete guards and control transitions:

```javascript
function validateTrajectory(body) {
  if (!body || !body.parameters) throw new Error("服务器返回的数据不完整");
  const arrays = [body.t, body.x, body.v];
  if (!arrays.every(Array.isArray) || arrays.some((values) => values.length === 0)) {
    throw new Error("服务器返回的轨迹为空");
  }
  if (body.t.length !== body.x.length || body.t.length !== body.v.length) {
    throw new Error("服务器返回的轨迹长度不一致");
  }
  if (!arrays.every((values) => values.every(Number.isFinite))) {
    throw new Error("服务器返回了无效数值");
  }
  const endpoint = body.t[body.t.length - 1];
  if (Math.abs(endpoint - body.parameters.t_end) > 1e-9) {
    throw new Error("服务器返回的终止时间不正确");
  }
}

function setLoading(loading) {
  state.loading = loading;
  elements["parameter-form"].querySelectorAll("input, button").forEach((control) => {
    control.disabled = loading;
  });
  updateControlState();
}

function updateControlState() {
  const unavailable = state.loading || !state.trajectory;
  elements["play-button"].disabled = unavailable || state.playing;
  elements["pause-button"].disabled = unavailable || !state.playing;
  elements["reset-button"].disabled = unavailable;
  elements["speed-select"].disabled = unavailable;
  elements["time-slider"].disabled = unavailable;
}
```

- [ ] **Step 6: Implement time interpolation and SVG mapping**

Use binary search so seeking works even though the final integration interval can differ from `0.01`:

```javascript
function sampleAt(time) {
  const { t, x, v } = state.trajectory;
  if (time <= t[0]) return { time: t[0], x: x[0], v: v[0] };
  if (time >= t[t.length - 1]) {
    const last = t.length - 1;
    return { time: t[last], x: x[last], v: v[last] };
  }
  let low = 0;
  let high = t.length - 1;
  while (high - low > 1) {
    const middle = Math.floor((low + high) / 2);
    if (t[middle] <= time) low = middle;
    else high = middle;
  }
  const ratio = (time - t[low]) / (t[high] - t[low]);
  return {
    time,
    x: x[low] + ratio * (x[high] - x[low]),
    v: v[low] + ratio * (v[high] - v[low]),
  };
}
```

Implement `springPath(startX, endX, y, coils=11, amplitude=22)` as an SVG path that starts with `M`, adds two short straight lead segments, alternates `y-amplitude` and `y+amplitude` for `coils*2` interior points, and ends at the mass edge. Guard the spring length with a minimum positive lead/interior span.

Use these rendering constants:

```javascript
const scene = { wallX: 80, centerX: 540, centerY: 166, massWidth: 112, massHeight: 96, travel: 255 };
const plot = { left: 70, right: 870, top: 32, bottom: 258 };
```

Compute `maxAbsX = Math.max(0.25, ...trajectory.x.map(Math.abs))`. Map the current displacement to `centerX + (sample.x / maxAbsX) * travel`, update `mass-block` coordinates, generate the spring path, and format live readouts with three decimals.

Generate the complete plot path once per trajectory. Map time linearly from `plot.left` to `plot.right`, map displacement symmetrically around the vertical midpoint, and update only `time-guide` and `current-point` per animation frame.

- [ ] **Step 7: Implement playback, seeking, and control transitions**

Use elapsed wall time rather than assumed frame duration:

```javascript
function animationFrame(timestamp) {
  if (!state.playing || !state.trajectory) return;
  if (state.lastFrameTime === null) state.lastFrameTime = timestamp;
  const elapsedSeconds = (timestamp - state.lastFrameTime) / 1000;
  state.lastFrameTime = timestamp;
  state.currentTime = Math.min(
    state.trajectory.parameters.t_end,
    state.currentTime + elapsedSeconds * state.speed,
  );
  elements["time-slider"].value = String(state.currentTime);
  renderCurrentState();
  if (state.currentTime >= state.trajectory.parameters.t_end) {
    pausePlayback();
    return;
  }
  state.animationFrameId = requestAnimationFrame(animationFrame);
}
```

Implement these exact transitions:

- `playPlayback()`: do nothing without a trajectory; reset time to zero when already at `t_end`; set `playing=true`, clear `lastFrameTime`, request a frame, update controls.
- `pausePlayback()`: cancel the stored animation frame if present, set `playing=false`, clear frame fields, update controls.
- `resetPlayback()`: pause, set time and slider to zero, render.
- Slider `input`: pause, parse and clamp its value, render immediately.
- Speed `change`: parse `0.5`, `1`, or `2` into `state.speed` without resetting time.
- Form `submit`: call `preventDefault()`, require `form.checkValidity()`, and call `requestTrajectory()`.
- Initial `DOMContentLoaded`: register all listeners and call `requestTrajectory()` so defaults render automatically.

`updateControlState()` disables play while already playing, disables pause while stopped, and disables all transport controls until a valid trajectory exists. `setLoading(true)` disables the parameter form and transport without discarding the last trajectory.

- [ ] **Step 8: Mount the frontend in FastAPI**

Modify `app.py` after constructing `app` and before endpoint definitions:

```python
from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


STATIC_DIR = Path(__file__).resolve().with_name("static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
```

Add the root endpoint:

```python
@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
```

- [ ] **Step 9: Run the frontend contract and full test suite**

Run:

```powershell
& 'D:\miniconda\envs\resnet\python.exe' -m unittest -v tests.test_frontend tests.test_browser_smoke
& 'D:\miniconda\envs\resnet\python.exe' -m unittest discover -s tests -v
```

Expected: frontend contract and real-browser smoke tests pass; all physics, API, and frontend tests pass without errors or warnings.

- [ ] **Step 10: Perform a headless-browser initial-render smoke test**

Start Uvicorn without opening a visible helper window:

```powershell
$server = Start-Process -FilePath 'D:\miniconda\envs\resnet\python.exe' `
  -ArgumentList '-m','uvicorn','app:app','--host','127.0.0.1','--port','8000' `
  -WorkingDirectory 'C:\codeee\pan' -WindowStyle Hidden -PassThru
```

Poll `http://127.0.0.1:8000/health` for at most ten seconds. Then run Chrome with a temporary user-data directory created by `New-Item`:

```powershell
& 'C:\Program Files\Google\Chrome\Application\chrome.exe' `
  --headless=new --disable-gpu --hide-scrollbars `
  --window-size=1440,1000 --virtual-time-budget=3000 `
  --user-data-dir="$tempChromeProfile" `
  --screenshot="$screenshotPath" 'http://127.0.0.1:8000/'
```

Inspect the screenshot and confirm that the default spring/mass state, curve, three readouts, control panel, and playback controls are visible without overlap. Stop only the exact `$server` process in `finally`, and remove only the verified temporary Chrome profile. If the visual check finds a layout defect, add a focused HTML-contract assertion where possible, watch it fail, fix CSS/markup, and rerun both tests and screenshot.

- [ ] **Step 11: Commit the browser interface**

```powershell
git add -- app.py static/index.html static/style.css static/app.js tests/test_frontend.py tests/test_browser_smoke.py
git commit -m "feat: add interactive spring simulator UI"
```

---

### Task 4: End-to-End Verification and Handoff

**Files:**
- Verify only: `spring_model.py`, `app.py`, `static/index.html`, `static/style.css`, `static/app.js`, `tests/test_spring_model.py`, `tests/test_api.py`, `tests/test_frontend.py`, `tests/test_browser_smoke.py`

**Interfaces:**
- Consumes: completed simulator from Tasks 1–3.
- Produces: verified localhost application and exact run instructions; no new product behavior.

- [ ] **Step 1: Run a clean full automated test pass**

```powershell
& 'D:\miniconda\envs\resnet\python.exe' -m unittest discover -s tests -v
```

Expected: every test reports `ok`, followed by `OK`; zero failures and zero errors.

- [ ] **Step 2: Check Python syntax compilation**

```powershell
& 'D:\miniconda\envs\resnet\python.exe' -m py_compile spring_model.py app.py tests/test_spring_model.py tests/test_api.py tests/test_frontend.py tests/test_browser_smoke.py
```

Expected: exit code `0` with no output.

- [ ] **Step 3: Start the real server and verify HTTP boundaries**

Start the server hidden as in Task 3, then make real HTTP requests with PowerShell:

```powershell
$health = Invoke-RestMethod 'http://127.0.0.1:8000/health'
$trajectory = Invoke-RestMethod -Method Post `
  -Uri 'http://127.0.0.1:8000/api/trajectory' `
  -ContentType 'application/json' `
  -Body '{"x0":-0.5018395,"v0":0.9391692,"t_end":10.0}'

if ($health.status -ne 'ok') { throw 'Health check failed' }
if ($trajectory.parameters.x0 -ne -0.502) { throw 'x0 normalization failed' }
if ($trajectory.parameters.v0 -ne 0.939) { throw 'v0 normalization failed' }
if ($trajectory.t.Count -ne $trajectory.x.Count -or $trajectory.t.Count -ne $trajectory.v.Count) { throw 'Trajectory array length mismatch' }
if ($trajectory.t[-1] -ne 10.0) { throw 'Trajectory endpoint mismatch' }
```

Expected: all assertions remain silent and the shell command exits `0`.

- [ ] **Step 4: Verify the browser interaction checklist**

Open `http://127.0.0.1:8000/` and verify each observable behavior:

1. Defaults load as `x0=-0.502`, `v0=0.939`, `t_end=10.0` and draw a spring, mass, and curve.
2. Play advances the mass, marker, slider, and `t/x/v` values together.
3. Pause freezes all animated state.
4. Dragging the slider seeks immediately and leaves playback paused.
5. Reset returns to `t=0` and the initial state.
6. `0.5×`, `1×`, and `2×` visibly change time progression without changing the physical trajectory.
7. Submitting new valid values recalculates and resets the visualization.
8. An out-of-range input is blocked or produces a visible validation message.
9. At `t_end`, playback stops; pressing Play again restarts from zero.
10. At a viewport narrower than `850px`, controls stack above the visualization without horizontal overflow.

Record any failure with its exact input and visible symptom. For a functional failure, write a failing regression test at the lowest testable boundary before changing implementation. For a purely visual defect, preserve a before screenshot, make the smallest CSS correction, and capture an after screenshot.

- [ ] **Step 5: Stop the exact server process and clean generated test artifacts**

Stop only the stored server PID. Remove verified temporary Chrome profiles, screenshots that are not needed for handoff, and Python `__pycache__` directories created by this feature. Do not remove or modify the user's CSV or existing Python files.

- [ ] **Step 6: Inspect the final diff and commit status**

```powershell
git status --short
git diff --check
git log -4 --oneline
```

Expected: no whitespace errors; only pre-existing unrelated user changes remain uncommitted; the feature has three focused commits for physics, API, and UI.

- [ ] **Step 7: Provide run instructions**

Handoff command:

```powershell
conda activate resnet
cd C:\codeee\pan
uvicorn app:app --reload
```

The user opens `http://127.0.0.1:8000/`. Report the automated test count, real API verification result, browser-check result, and any remaining limitations. Do not claim completion without fresh output from Steps 1–4.
