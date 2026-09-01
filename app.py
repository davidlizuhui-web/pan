from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from spring_model import MASS, OMEGA, PERIOD, SPRING_CONSTANT, solve_trajectory


class TrajectoryRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    x0: float = Field(ge=-2.0, le=2.0)
    v0: float = Field(ge=-1.0, le=1.0)
    t_end: float = Field(ge=0.1, le=60.0)


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
