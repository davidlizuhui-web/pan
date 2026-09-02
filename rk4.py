import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 基本参数
# =========================

m = 1.0
k = np.pi**2
omega = np.sqrt(k / m)

num_trajectories = 50

t_start = 0.0
t_end = 10.0
num_time_steps = 1000

t = np.linspace(t_start, t_end, num_time_steps)

noise_std = 0.001

# 固定随机种子，使结果可重复
np.random.seed(42)


# =========================
# 2. 随机生成初始条件
# =========================

x0_list = np.round(
    np.random.uniform(
        low=-2.0,
        high=2.0,
        size=num_trajectories,
    ),
    decimals=3,
)

v0_list = np.round(
    np.random.uniform(
        low=-1.0,
        high=1.0,
        size=num_trajectories,
    ),
    decimals=3,
)


# =========================
# 3. 定义动力学方程
# =========================

def derivatives(x, v):
    """
    弹簧振子的状态方程：

        dx/dt = v
        dv/dt = -(k/m)x = -omega^2 x
    """
    dx_dt = v
    dv_dt = -(omega**2) * x

    return dx_dt, dv_dt


# =========================
# 4. 单步四阶 Runge-Kutta
# =========================

def rk4_step(x, v, dt):
    # k1
    k1_x, k1_v = derivatives(x, v)

    # k2
    k2_x, k2_v = derivatives(
        x + 0.5 * dt * k1_x,
        v + 0.5 * dt * k1_v,
    )

    # k3
    k3_x, k3_v = derivatives(
        x + 0.5 * dt * k2_x,
        v + 0.5 * dt * k2_v,
    )

    # k4
    k4_x, k4_v = derivatives(
        x + dt * k3_x,  
        v + dt * k3_v,
    )

    x_next = x + (dt / 6.0) * (
        k1_x
        + 2.0 * k2_x
        + 2.0 * k3_x
        + k4_x
    )

    v_next = v + (dt / 6.0) * (
        k1_v
        + 2.0 * k2_v
        + 2.0 * k3_v
        + k4_v
    )

    return x_next, v_next


# =========================
# 5. 使用 RK4 生成轨迹
# =========================

trajectories = []

for trajectory_id in range(num_trajectories):
    x0 = x0_list[trajectory_id]
    v0 = v0_list[trajectory_id]

    # 无噪声的数值解
    x_clean = np.zeros(num_time_steps)
    v_clean = np.zeros(num_time_steps)

    # 初始条件
    x_clean[0] = x0
    v_clean[0] = v0

    # RK4 时间积分
    for n in range(num_time_steps - 1):
        dt = t[n + 1] - t[n]

        x_clean[n + 1], v_clean[n + 1] = rk4_step(
            x_clean[n],
            v_clean[n],
            dt,
        )

    # 分别生成独立的高斯观测噪声
    x_noise = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=num_time_steps,
    )

    v_noise = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=num_time_steps,
    )

    # 加噪后的观测数据
    x_noisy = x_clean + x_noise
    v_noisy = v_clean + v_noise

    trajectory = {
        "trajectory_id": trajectory_id,
        "x0": x0,
        "v0": v0,
        "t": t,
        "x_clean": x_clean,
        "v_clean": v_clean,
        "x": x_noisy,
        "v": v_noisy,
    }

    trajectories.append(trajectory)


def save_trajectories_to_csv(trajectories, csv_path):
    fieldnames = [
        "trajectory_id",
        "x0",
        "v0",
        "t",
        "x_clean",
        "v_clean",
        "x",
        "v",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for trajectory in trajectories:
            for index in range(len(trajectory["t"])):
                writer.writerow(
                    {
                        "trajectory_id": trajectory["trajectory_id"],
                        "x0": trajectory["x0"],
                        "v0": trajectory["v0"],
                        "t": trajectory["t"][index],
                        "x_clean": trajectory["x_clean"][index],
                        "v_clean": trajectory["v_clean"][index],
                        "x": trajectory["x"][index],
                        "v": trajectory["v"][index],
                    }
                )


csv_path = Path(__file__).resolve().with_name("rk4_trajectories.csv")
save_trajectories_to_csv(trajectories, csv_path)


# =========================
# 6. 输出基本信息
# =========================

print(f"生成轨迹数量: {len(trajectories)}")
print(f"omega = {omega:.4f}")
print(f"period = {2 * np.pi / omega:.4f} s")
print(f"noise standard deviation = {noise_std}")
print(f"CSV saved to: {csv_path}")
print(f"CSV data rows: {num_trajectories * num_time_steps}")

print("\n第一条轨迹:")
print(f"x0 = {trajectories[0]['x0']:.4f}")
print(f"v0 = {trajectories[0]['v0']:.4f}")
print(f"x shape = {trajectories[0]['x'].shape}")
print(f"v shape = {trajectories[0]['v'].shape}")


# =========================
# 7. 绘制带噪位置轨迹
# =========================

plt.figure(figsize=(12, 6))

for trajectory in trajectories:
    plt.plot(
        trajectory["t"],
        trajectory["x"],
        alpha=0.6,
    )

plt.xlabel("Time t")
plt.ylabel("Position x(t)")
plt.title("50 RK4 Mass-Spring Trajectories with Gaussian Noise")
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================
# 8. 绘制带噪相空间
# =========================

plt.figure(figsize=(8, 8))

for trajectory in trajectories:
    plt.plot(
        trajectory["x"],
        trajectory["v"],
        alpha=0.6,
    )

plt.xlabel("Position x")
plt.ylabel("Velocity v")
plt.title("RK4 Phase Space with Gaussian Noise")
plt.grid(True)
plt.tight_layout()
plt.show()
