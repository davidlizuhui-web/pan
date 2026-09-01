"use strict";

const state = {
  trajectory: null,
  currentTime: 0,
  playing: false,
  speed: 1,
  loading: false,
  lastFrameTime: null,
  animationFrameId: null,
  maxAbsX: 1,
};

const scene = {
  wallX: 80,
  centerX: 540,
  centerY: 166,
  massWidth: 112,
  massHeight: 96,
  travel: 255,
};
const plot = { left: 70, right: 870, top: 32, bottom: 258 };

const elementIds = [
  "parameter-form", "x0-input", "v0-input", "duration-input",
  "calculate-button", "play-button", "pause-button", "reset-button",
  "speed-select", "time-slider", "time-value", "position-value",
  "velocity-value", "error-message", "spring-path", "mass-block",
  "mass-label", "trajectory-path", "time-guide", "current-point",
  "timeline-end",
];
const elements = Object.fromEntries(
  elementIds.map((id) => [id, document.getElementById(id)]),
);

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function validationMessage(body) {
  if (Array.isArray(body?.detail) && body.detail.length > 0) {
    return body.detail[0]?.msg || "输入参数无效";
  }
  return "请求失败，请检查输入参数";
}

function validateTrajectory(body) {
  if (!body || !body.parameters) {
    throw new Error("服务器返回的数据不完整");
  }
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

function showError(message) {
  elements["error-message"].textContent = message;
}

function updateControlState() {
  const unavailable = state.loading || !state.trajectory;
  elements["play-button"].disabled = unavailable || state.playing;
  elements["pause-button"].disabled = unavailable || !state.playing;
  elements["reset-button"].disabled = unavailable;
  elements["speed-select"].disabled = unavailable;
  elements["time-slider"].disabled = unavailable;
}

function setLoading(loading) {
  state.loading = loading;
  elements["parameter-form"].querySelectorAll("input, button").forEach((control) => {
    control.disabled = loading;
  });
  elements["calculate-button"].querySelector("span").textContent = loading
    ? "计算中…"
    : "计算轨迹";
  updateControlState();
}

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
    if (!response.ok) {
      throw new Error(validationMessage(body));
    }
    validateTrajectory(body);
    state.trajectory = body;
    state.currentTime = 0;
    state.maxAbsX = Math.max(0.25, ...body.x.map(Math.abs));
    elements["x0-input"].value = body.parameters.x0.toFixed(3);
    elements["v0-input"].value = body.parameters.v0.toFixed(3);
    elements["time-slider"].max = String(body.parameters.t_end);
    elements["time-slider"].value = "0";
    elements["timeline-end"].textContent = `${body.parameters.t_end.toFixed(1)} s`;
    renderTrajectoryPath();
    renderCurrentState();
  } catch (error) {
    showError(error instanceof Error ? error.message : "无法计算轨迹");
  } finally {
    setLoading(false);
  }
}

function sampleAt(time) {
  const { t, x, v } = state.trajectory;
  if (time <= t[0]) {
    return { time: t[0], x: x[0], v: v[0] };
  }
  if (time >= t[t.length - 1]) {
    const last = t.length - 1;
    return { time: t[last], x: x[last], v: v[last] };
  }

  let low = 0;
  let high = t.length - 1;
  while (high - low > 1) {
    const middle = Math.floor((low + high) / 2);
    if (t[middle] <= time) {
      low = middle;
    } else {
      high = middle;
    }
  }

  const ratio = (time - t[low]) / (t[high] - t[low]);
  return {
    time,
    x: x[low] + ratio * (x[high] - x[low]),
    v: v[low] + ratio * (v[high] - v[low]),
  };
}

function springPath(startX, endX, y, coils = 11, amplitude = 22) {
  const safeEnd = Math.max(startX + 48, endX);
  const lead = Math.min(34, (safeEnd - startX) * 0.16);
  const innerStart = startX + lead;
  const innerEnd = safeEnd - lead;
  const points = [`M ${startX.toFixed(2)} ${y}`, `L ${innerStart.toFixed(2)} ${y}`];
  const zigzags = coils * 2;

  for (let index = 1; index < zigzags; index += 1) {
    const x = innerStart + ((innerEnd - innerStart) * index) / zigzags;
    const offset = index % 2 === 0 ? -amplitude : amplitude;
    points.push(`L ${x.toFixed(2)} ${(y + offset).toFixed(2)}`);
  }

  points.push(`L ${innerEnd.toFixed(2)} ${y}`, `L ${safeEnd.toFixed(2)} ${y}`);
  return points.join(" ");
}

function plotCoordinates(time, position) {
  const duration = state.trajectory.parameters.t_end;
  const x = plot.left + (time / duration) * (plot.right - plot.left);
  const midpoint = (plot.top + plot.bottom) / 2;
  const y = midpoint - (position / state.maxAbsX) * ((plot.bottom - plot.top) / 2);
  return { x, y };
}

function renderTrajectoryPath() {
  const { t, x } = state.trajectory;
  const path = t.map((time, index) => {
    const point = plotCoordinates(time, x[index]);
    return `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`;
  }).join(" ");
  elements["trajectory-path"].setAttribute("d", path);
}

function renderCurrentState() {
  if (!state.trajectory) {
    return;
  }
  const sample = sampleAt(state.currentTime);
  const massCenterX = scene.centerX + (sample.x / state.maxAbsX) * scene.travel;
  const massLeft = massCenterX - scene.massWidth / 2;
  const massTop = scene.centerY - scene.massHeight / 2;
  const springStart = scene.wallX;

  elements["mass-block"].setAttribute("x", massLeft.toFixed(2));
  elements["mass-block"].setAttribute("y", massTop.toFixed(2));
  elements["mass-label"].setAttribute("x", massCenterX.toFixed(2));
  elements["mass-label"].setAttribute("y", (scene.centerY + 10).toFixed(2));
  elements["spring-path"].setAttribute(
    "d",
    springPath(springStart, massLeft, scene.centerY),
  );

  elements["time-value"].textContent = sample.time.toFixed(3);
  elements["position-value"].textContent = sample.x.toFixed(3);
  elements["velocity-value"].textContent = sample.v.toFixed(3);

  const point = plotCoordinates(sample.time, sample.x);
  elements["time-guide"].setAttribute("x1", point.x.toFixed(2));
  elements["time-guide"].setAttribute("x2", point.x.toFixed(2));
  elements["time-guide"].setAttribute("y1", String(plot.top));
  elements["time-guide"].setAttribute("y2", String(plot.bottom));
  elements["current-point"].setAttribute("cx", point.x.toFixed(2));
  elements["current-point"].setAttribute("cy", point.y.toFixed(2));
}

function animationFrame(timestamp) {
  if (!state.playing || !state.trajectory) {
    return;
  }
  if (state.lastFrameTime === null) {
    state.lastFrameTime = timestamp;
  }
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

function playPlayback() {
  if (!state.trajectory || state.playing) {
    return;
  }
  if (state.currentTime >= state.trajectory.parameters.t_end) {
    state.currentTime = 0;
    elements["time-slider"].value = "0";
    renderCurrentState();
  }
  state.playing = true;
  state.lastFrameTime = null;
  state.animationFrameId = requestAnimationFrame(animationFrame);
  updateControlState();
}

function pausePlayback() {
  if (state.animationFrameId !== null) {
    cancelAnimationFrame(state.animationFrameId);
  }
  state.playing = false;
  state.animationFrameId = null;
  state.lastFrameTime = null;
  updateControlState();
}

function resetPlayback() {
  pausePlayback();
  state.currentTime = 0;
  elements["time-slider"].value = "0";
  renderCurrentState();
}

function initialize() {
  elements["parameter-form"].addEventListener("submit", (event) => {
    event.preventDefault();
    if (elements["parameter-form"].reportValidity()) {
      requestTrajectory();
    }
  });
  elements["play-button"].addEventListener("click", playPlayback);
  elements["pause-button"].addEventListener("click", pausePlayback);
  elements["reset-button"].addEventListener("click", resetPlayback);
  elements["speed-select"].addEventListener("change", () => {
    state.speed = Number(elements["speed-select"].value);
  });
  elements["time-slider"].addEventListener("input", () => {
    pausePlayback();
    state.currentTime = clamp(
      Number(elements["time-slider"].value),
      0,
      state.trajectory?.parameters.t_end ?? 0,
    );
    renderCurrentState();
  });

  updateControlState();
  requestTrajectory();
}

document.addEventListener("DOMContentLoaded", initialize);
