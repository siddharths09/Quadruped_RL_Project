import numpy as np
import matplotlib.pyplot as plt

# ---- EDIT THESE ----
npz_with_gait = "/home/caitlinregan/quad_project/Quadruped_RL_Project/left_foot_tracking_eval_with_gait.npz"
npz_without_gait = "/home/caitlinregan/quad_project/Quadruped_RL_Project/left_foot_tracking_eval.npz"

# If you know your control dt (seconds per step), set it; otherwise it just uses "steps"
DT = None  # e.g., 0.02

def load_traj(path):
    d = np.load(path)
    x = d["fl_foot_x"].astype(float)
    y = d["fl_foot_y"].astype(float)
    z = d["fl_foot_z"].astype(float)
    n = len(z)
    t = np.arange(n) if DT is None else np.arange(n) * DT
    t_label = "step" if DT is None else "time (s)"
    return t, x, y, z, t_label

def moving_min(a, k=25):
    # simple rolling min, edges handled by padding
    if k <= 1:
        return a.copy()
    pad = k // 2
    ap = np.pad(a, (pad, pad), mode="edge")
    out = np.empty_like(a)
    for i in range(len(a)):
        out[i] = ap[i:i+k].min()
    return out

def step_metrics(t, z):
    """
    Extract step-ish metrics from z(t) only:
    - estimate ground height using rolling minimum
    - define swing when z > ground + clearance
    - find swing peaks -> swing height
    - estimate step intervals from successive peaks
    """
    # Estimate "ground" as rolling min (works decently if the foot touches the ground often)
    z_ground = moving_min(z, k=51)
    # Clearance threshold: a little above estimated ground (tweak if needed)
    clearance = 0.02  # meters-ish; adjust if your scale differs
    swing = z > (z_ground + clearance)

    # Find swing peaks: local maxima during swing
    peaks = []
    for i in range(1, len(z)-1):
        if swing[i] and z[i] >= z[i-1] and z[i] >= z[i+1]:
            # require it to be a reasonably "prominent" peak above ground
            if z[i] - z_ground[i] > clearance:
                peaks.append(i)
    peaks = np.array(peaks, dtype=int)

    # Reduce duplicate peaks that happen on flat tops by keeping spaced ones
    if len(peaks) > 1:
        keep = [peaks[0]]
        min_sep = 10  # steps; increase if you see too many peaks
        for p in peaks[1:]:
            if p - keep[-1] >= min_sep:
                keep.append(p)
        peaks = np.array(keep, dtype=int)

    swing_heights = (z[peaks] - z_ground[peaks]) if len(peaks) else np.array([])
    step_intervals = np.diff(t[peaks]) if len(peaks) > 1 else np.array([])

    # Dragging proxy: fraction of time foot is very close to ground
    near_ground = (z - z_ground) < 0.005  # 5 mm-ish
    drag_frac = near_ground.mean()

    return {
        "z_ground": z_ground,
        "swing_mask": swing,
        "peaks": peaks,
        "swing_heights": swing_heights,
        "step_intervals": step_intervals,
        "drag_frac": drag_frac,
    }

# Load both
tA, xA, yA, zA, t_label = load_traj(npz_with_gait)
tB, xB, yB, zB, _      = load_traj(npz_without_gait)

mA = step_metrics(tA, zA)
mB = step_metrics(tB, zB)

# ---- PLOTS ----

# 1) Z trajectory over time + peaks
plt.figure()
plt.plot(tA, zA, label="With gait-shaping")
plt.plot(tB, zB, label="Without gait-shaping")
plt.scatter(tA[mA["peaks"]], zA[mA["peaks"]], s=10)
plt.scatter(tB[mB["peaks"]], zB[mB["peaks"]], s=10)
plt.xlabel(t_label)
plt.ylabel("FL foot z (world)")
plt.title("Front Left Foot Position Trajectory")
plt.legend()
plt.show()

# 2) XY trajectory (foot placement path)
plt.figure()
plt.plot(xA, yA, label="With gait-shaping")
plt.plot(xB, yB, label="Without gait-shaping")
plt.xlabel("FL foot x (world)")
plt.ylabel("FL foot y (world)")
plt.title("Front-left foot XY path")
plt.axis("equal")
plt.legend()
plt.show()

# 3) Swing height distributions
plt.figure()
bins = 30
plt.hist(mA["swing_heights"], bins=bins, alpha=0.6, label="With gait-shaping")
plt.hist(mB["swing_heights"], bins=bins, alpha=0.6, label="Without gait-shaping")
plt.xlabel("Swing height above estimated ground")
plt.ylabel("count")
plt.title("Swing height distribution (clearance consistency)")
plt.legend()
plt.show()

# 4) Step timing distributions (interval between swing peaks)
plt.figure()
plt.hist(mA["step_intervals"], bins=bins, alpha=0.6, label="With gait-shaping")
plt.hist(mB["step_intervals"], bins=bins, alpha=0.6, label="Without gait-shaping")
plt.xlabel(f"Step interval ({t_label})")
plt.ylabel("count")
plt.title("Step timing regularity (peak-to-peak intervals)")
plt.legend()
plt.show()

# ---- QUICK SUMMARY PRINT ----
def summarize(tag, metrics):
    sh = metrics["swing_heights"]
    si = metrics["step_intervals"]
    print(f"\n{tag}")
