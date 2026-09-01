# Interactive Mass-Spring Simulator Design

## Objective

Build a browser-based interactive simulator for the existing ideal mass-spring model. The user supplies the initial displacement `x0`, initial velocity `v0`, and simulation duration `t_end`. Python computes the trajectory with the existing fourth-order Runge-Kutta method, while JavaScript renders and controls the animation in the browser.

The application runs in the Conda `resnet` environment and uses FastAPI, NumPy, Uvicorn, and native HTML/CSS/JavaScript. These runtime dependencies are already present in that environment.

## Scope

The first version will provide:

- Inputs for `x0`, `v0`, and `t_end`.
- RK4 trajectory calculation in Python.
- A one-dimensional wall, spring, and moving mass visualization.
- A synchronized displacement-versus-time plot.
- A time slider plus play, pause, reset, and playback-speed controls.
- Live display of the current time, position, and velocity.
- Input validation and visible error feedback.

The first version will not include damping, external forcing, editable mass or spring stiffness, Gaussian observation noise, authentication, persistent storage, or deployment configuration.

## Physical Model

The simulator uses the existing undamped, unforced oscillator:

\[
m\ddot{x}+kx=0
\]

with fixed parameters:

\[
m=1,\qquad k=\pi^2,\qquad \omega=\sqrt{k/m}=\pi.
\]

The first-order system integrated by RK4 is:

\[
\frac{dx}{dt}=v,\qquad \frac{dv}{dt}=-\omega^2x.
\]

Initial values are rounded to three decimal places before integration, matching the existing data-generation convention. The solver uses a target step size of `0.01` seconds and always includes the exact requested endpoint. The API returns clean physical state only; measurement noise is outside this version's scope.

## Architecture

The application separates physics, HTTP handling, and presentation:

```text
Browser controls
      |
      | POST /api/trajectory
      v
FastAPI request validation
      |
      v
Python RK4 solver
      |
      | JSON: parameters, t[], x[], v[]
      v
Browser animation state
      |
      +--> spring and mass SVG
      +--> displacement plot SVG
      +--> time slider and live values
```

Python computes a complete trajectory only when the page loads or the user submits new parameters. Playback does not make a request for every frame. JavaScript stores the returned trajectory, advances simulation time with `requestAnimationFrame`, interpolates between adjacent samples, and updates the visualization locally.

## Planned Files

```text
spring_model.py
app.py
static/
  index.html
  style.css
  app.js
tests/
  test_spring_model.py
  test_api.py
```

- `spring_model.py` owns the equations, one RK4 step, and complete-trajectory calculation.
- `app.py` owns FastAPI request and response models, the trajectory endpoint, the index route, and static-file mounting.
- `static/index.html` contains semantic page structure and controls.
- `static/style.css` contains the responsive layout and simulation styling.
- `static/app.js` owns API requests, playback state, interpolation, SVG rendering, and user-visible errors.
- `tests/test_spring_model.py` verifies numerical behavior independently of HTTP.
- `tests/test_api.py` verifies the API contract, validation, and frontend availability.

The existing `rk4.py` remains a standalone CSV and plotting workflow. Its solver behavior will not be coupled to web-server startup or frontend state.

## API Contract

### `POST /api/trajectory`

Request body:

```json
{
  "x0": -0.502,
  "v0": 0.939,
  "t_end": 10.0
}
```

Validation rules:

- `x0` must be finite and between `-2.0` and `2.0`.
- `v0` must be finite and between `-1.0` and `1.0`.
- `t_end` must be finite and between `0.1` and `60.0` seconds.
- `x0` and `v0` are rounded to three decimal places on the server, so the server remains authoritative even if a non-browser client sends more precision.

Successful response:

```json
{
  "parameters": {
    "x0": -0.502,
    "v0": 0.939,
    "t_end": 10.0,
    "m": 1.0,
    "k": 9.869604401089358,
    "omega": 3.141592653589793,
    "period": 2.0
  },
  "t": [0.0, 0.01],
  "x": [-0.502, -0.492364],
  "v": [0.939, 0.988]
}
```

The shown arrays are abbreviated examples. In the real response, `t`, `x`, and `v` have equal nonzero lengths, begin at the requested initial state, and end at `t_end`.

Invalid requests use FastAPI's standard `422` validation response. Unexpected calculation failures return a generic server error without exposing a traceback to the page.

### Frontend routes

- `GET /` returns `static/index.html`.
- `GET /static/style.css` and `GET /static/app.js` return frontend assets.
- `GET /health` returns a small status response for startup verification.

## User Interface

The desktop layout has a compact control panel on the left and the simulation on the right. On narrow screens the control panel stacks above the simulation.

Controls:

- Numeric `x0` input with step `0.001` and range `[-2, 2]`.
- Numeric `v0` input with step `0.001` and range `[-1, 1]`.
- Numeric duration input with step `0.1` and range `[0.1, 60]`.
- A primary "Calculate trajectory" button.
- Play, pause, and reset buttons.
- Playback speed choices `0.5x`, `1x`, and `2x`.
- A time slider covering `[0, t_end]`.

Defaults are `x0=-0.502`, `v0=0.939`, and `t_end=10.0`.

The simulation SVG contains a fixed wall, a dynamically generated zigzag spring, a mass block, an equilibrium marker, and a horizontal guide. The visualization scales the model's displacement range into the available SVG width with padding; it does not alter the numerical values shown to the user.

The displacement plot is another SVG. It shows the complete `x(t)` curve, a vertical current-time guide, and a moving point at the current state. The current `t`, `x`, and `v` values update together with the mass position.

## Interaction and State

The frontend has four playback states:

- `empty`: no valid trajectory is available.
- `ready`: a trajectory is loaded at a selected time.
- `playing`: animation time advances.
- `error`: the last request failed and an error message is visible.

Submitting parameters pauses any active playback, disables controls while the request is pending, fetches a new trajectory, resets the current time to zero, and enters `ready` on success. On failure, the last valid visualization remains visible where possible and the interface enters `error`.

Play starts or resumes from the current slider time. If play is pressed at the endpoint, playback restarts at zero. Pause freezes the current time. Reset pauses and returns to zero. Dragging the time slider pauses playback and renders the selected state immediately. Playback stops in `ready` at `t_end`.

`requestAnimationFrame` supplies wall-clock timestamps. Simulation time advances by elapsed wall time multiplied by the selected speed, avoiding drift caused by assuming a fixed browser frame rate.

## Error Handling

- Browser inputs use HTML constraints for immediate feedback.
- FastAPI independently validates all inputs and remains authoritative.
- The frontend checks HTTP status and the returned array shapes before accepting a trajectory.
- While a request is running, duplicate submissions and playback are disabled.
- Network and validation errors appear in an accessible message region.
- JavaScript stops playback if it encounters non-finite trajectory values.

## Testing and Verification

Tests use the standard-library `unittest` runner because `pytest` is not installed in `resnet`.

Physics tests cover:

- Exact preservation of `x(0)=x0` and `v(0)=v0`.
- Inclusion of the exact requested endpoint.
- Equal lengths for `t`, `x`, and `v`.
- Three-decimal normalization of initial values.
- RK4 agreement with the analytical oscillator solution within a stated tolerance.
- Rejection of invalid duration or step size at the model boundary.

API tests cover:

- Successful response schema and initial values.
- Validation failures for out-of-range and non-finite parameters.
- Health endpoint behavior.
- Availability of the index page and static assets.

Completion verification runs all tests with the `resnet` Python executable, starts Uvicorn in that environment, requests the health endpoint and trajectory API, and inspects the page in a browser. Playback, pause, reset, slider seeking, recalculation, responsive layout, and visible error handling receive a manual browser check.

## Running the Application

From the workspace in the `resnet` environment:

```powershell
conda activate resnet
uvicorn app:app --reload
```

The browser then opens `http://127.0.0.1:8000/`. The application binds to localhost by default and does not modify CSV files or start automatically when imported by tests.
