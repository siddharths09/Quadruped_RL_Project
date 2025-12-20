import numpy as np
import matplotlib.pyplot as plt

data = np.load("/home/caitlinregan/mjlab/velocity_tracking_eval_forward.npz")

N = 125
steps = np.arange(N)

# ---------- Linear velocity X ----------
plt.figure()
plt.plot(steps, data["cmd_vx"][:N], label="Commanded $v_x$")
plt.plot(steps, data["act_vx"][:N], label="Actual $v_x$")
plt.xlabel("Environment step")
plt.ylabel("Linear velocity [m/s]")
plt.title("Forward velocity tracking (125 steps)")
plt.legend()
plt.grid(True)

# ---------- Linear velocity Y ----------
plt.figure()
plt.plot(steps, data["cmd_vy"][:N], label="Commanded $v_y$")
plt.plot(steps, data["act_vy"][:N], label="Actual $v_y$")
plt.xlabel("Environment step")
plt.ylabel("Linear velocity [m/s]")
plt.title("Lateral velocity tracking (125 steps)")
plt.legend()
plt.grid(True)

# ---------- Angular velocity Z ----------
plt.figure()
plt.plot(steps, data["cmd_wz"][:N], label="Commanded $\omega_z$")
plt.plot(steps, data["act_wz"][:N], label="Actual $\omega_z$")
plt.xlabel("Environment step")
plt.ylabel("Angular velocity [rad/s]")
plt.title("Yaw velocity tracking (125 steps)")
plt.legend()
plt.grid(True)

plt.show()
