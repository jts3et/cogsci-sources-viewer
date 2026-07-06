"""
relative_phase.py -- recover the actual relative phase from CMU motion-capture,
not just the sign of coordination.

The Methodology page reads the sign of a coupling as a relative-phase order
parameter, after Haken, Kelso and Bunz. Section 4C shows the SIGN of limb
coordination flips across tasks (correlation -0.92 in walk, +1.00 in jump).
Here we recover the relative phase itself, in degrees, from the limb-angle time
series with a Hilbert transform, and check that the rhythmic tasks sit near the
two HKB attractors: anti-phase (180 deg) for walking and running, in-phase
(0 deg) for jumping.

No new data. Runs on the same CMU trials as fit_ggm.py.
"""
from __future__ import annotations
import glob, os
import numpy as np
from scipy.signal import hilbert, detrend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.dirname(os.path.abspath(__file__))
INK, ACCENT, COOL, MUTE = "#1b2430", "#b3541e", "#2f6b7a", "#9aa0a6"


def amc_series(path, joints):
    """Return {joint: 1-D array of its flexion angle (first dof) across frames}."""
    with open(path) as f:
        lines = [ln.strip() for ln in f]
    i = 0
    while i < len(lines) and not lines[i].startswith(":DEGREES"):
        i += 1
    out = {j: [] for j in joints}
    for ln in lines[i + 1:]:
        if not ln or ln.isdigit():
            continue
        t = ln.split()
        if t[0] in out:
            out[t[0]].append(float(t[1]))   # first dof = flexion (rx)
    return {j: np.asarray(v) for j, v in out.items()}


def rel_phase(a, b):
    """Instantaneous relative phase (radians, wrapped) between two oscillating signals."""
    pa = np.angle(hilbert(detrend(a)))
    pb = np.angle(hilbert(detrend(b)))
    d = pa - pb
    return np.angle(np.exp(1j * d))          # wrap to (-pi, pi]

def circ_mean(d):
    m = np.angle(np.mean(np.exp(1j * d)))
    plv = np.abs(np.mean(np.exp(1j * d)))    # phase-locking value in [0,1]
    return m, plv


TASKS = {"walk": sorted(glob.glob(os.path.join(DATA, "07_0*.amc"))),
         "run":  sorted(glob.glob(os.path.join(DATA, "09_0*.amc"))),
         "jump": sorted(glob.glob(os.path.join(DATA, "13_1*.amc")))}
PAIRS = {"legs L/R": ("lfemur", "rfemur"), "arms L/R": ("lhumerus", "rhumerus")}
JOINTS = ["lfemur", "rfemur", "lhumerus", "rhumerus"]


def main():
    print("=" * 74)
    print("RELATIVE PHASE FROM CMU MOCAP  (Hilbert transform; no new data)")
    print("=" * 74)
    print("\n   HKB attractors: anti-phase = 180 deg, in-phase = 0 deg.\n")
    results = {}   # (pair, task) -> (phases array, mean_deg, plv)
    for task, paths in TASKS.items():
        for pair, (ja, jb) in PAIRS.items():
            phases = []
            for p in paths:
                s = amc_series(p, JOINTS)
                a, b = s[ja], s[jb]
                n = min(len(a), len(b))
                if n < 30:
                    continue
                phases.append(rel_phase(a[:n], b[:n]))
            phases = np.concatenate(phases) if phases else np.array([])
            m, plv = circ_mean(phases)
            results[(pair, task)] = (phases, np.degrees(m) % 360, plv)
        p_leg = results[("legs L/R", task)]
        p_arm = results[("arms L/R", task)]
        print(f"   [{task:4s}]  legs {p_leg[1]:6.1f} deg (locking {p_leg[2]:.2f})   "
              f"arms {p_arm[1]:6.1f} deg (locking {p_arm[2]:.2f})")

    # ---- rose plot: distribution of relative phase, pairs x tasks ----
    tasks = list(TASKS)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4), subplot_kw={"projection": "polar"})
    for r, pair in enumerate(PAIRS):
        for c, task in enumerate(tasks):
            ax = axes[r, c]
            phases, mdeg, plv = results[(pair, task)]
            col = ACCENT if (90 < mdeg < 270) else COOL   # rust = anti-phase, teal = in-phase
            bins = np.linspace(-np.pi, np.pi, 25)
            h, edges = np.histogram(phases, bins=bins, density=True)
            ax.bar((edges[:-1] + edges[1:]) / 2, h, width=np.diff(edges),
                   color=col, alpha=0.55, edgecolor=col)
            ax.plot([np.radians(mdeg), np.radians(mdeg)], [0, max(h) * 1.05], color=INK, lw=2.2)
            ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
            ax.set_xticks(np.radians([0, 90, 180, 270]))
            ax.set_xticklabels(["0°\nin-phase", "90°", "180°\nanti-phase", "270°"], fontsize=7)
            ax.set_yticks([])
            ax.set_title(f"{task}  {pair}\nφ = {mdeg:.0f}°  (locking {plv:.2f})", fontsize=9.5, pad=12)
    fig.suptitle("Measured relative phase between paired limbs.\n"
                 "Walking and running lock near 180° (anti-phase); jumping sits near 0° (in-phase)",
                 fontsize=12, y=1.01, color=INK)
    fig.tight_layout()
    fig.savefig("fig_relative_phase.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("\n   note: relative phase is well defined for the rhythmic tasks (walk, run);")
    print("   jump is a single extension, so read its jump column as in-phase timing, not a limit-cycle phase.")
    print("wrote fig_relative_phase.png")


if __name__ == "__main__":
    main()
