import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 基本参数
# =========================
#高斯噪声 拟合 观测噪声
noise_std = 0.005

m = 1.0
k = np.pi**2

omega = np.sqrt(k / m)

num_trajectories = 50

t_start = 0.0
t_end = 10.0
num_time_steps = 1000

# 时间点
t = np.linspace(t_start, t_end, num_time_steps)

# 固定随机种子，保证每次运行结果一致
np.random.seed(42)



# =========================
# 2. 随机生成初始条件
# =========================

# x0 ∈ [-2, 2]
x0_list = np.random.uniform(low=-2.0, high=2.0, size=num_trajectories)

# v0 ∈ [-1, 1]
v0_list = np.random.uniform(low=-1.0, high=1.0, size=num_trajectories)


# =========================
# 3. 生成轨迹
# =========================

trajectories = []

for i in range(num_trajectories):
    # 高斯噪声
    x_noise = np.random.normal(0.0, noise_std, size=t.shape)
    v_noise = np.random.normal(0.0, noise_std, size=t.shape)

    x0 = x0_list[i]
    v0 = v0_list[i]

    # 解析解：位置
    x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t) +x_noise

    # 解析解：速度
    v = -omega * x0 * np.sin(omega * t) + v0 * np.cos(omega * t) +v_noise

    trajectory = {
        "trajectory_id": i,
        "x0": x0,
        "v0": v0,
        "t": t,
        "x": x,
        "v": v,
    }

    trajectories.append(trajectory)


print(f"生成轨迹数量: {len(trajectories)}")
print(f"omega = {omega:.4f}")
print(f"period = {2 * np.pi / omega:.4f} s")

print("\n第一条轨迹:")
print(f"x0 = {trajectories[0]['x0']:.4f}")
print(f"v0 = {trajectories[0]['v0']:.4f}")
print(f"x shape = {trajectories[0]['x'].shape}")
print(f"v shape = {trajectories[0]['v'].shape}")


# =========================
# 4. 绘制 50 条位置轨迹
# =========================

plt.figure(figsize=(12, 6))

for trajectory in trajectories:
    plt.plot(trajectory["t"], trajectory["x"], alpha=0.6)

plt.xlabel("Time t")
plt.ylabel("Position x(t)")
plt.title("50 Mass-Spring Trajectories")
plt.grid(True)

plt.show()


# =========================
# 5. 绘制 phase space
# =========================

plt.figure(figsize=(8, 8))

for trajectory in trajectories:
    plt.plot(trajectory["x"], trajectory["v"], alpha=0.6)

plt.xlabel("Position x")
plt.ylabel("Velocity v")
plt.title("Phase Space")
plt.grid(True)

plt.show()
