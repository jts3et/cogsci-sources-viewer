"""
two_body.py -- STEP 1 of the intended outcome: two-body convergence.

Claim to test (from the Methodology page, section 6): a metaphor drives two
differently built bodies onto the same relative-phase coordination mode, each
body reaching it in its own coordinates, whereas a joint-angle instruction stays
specific to one body and cannot align their coordination at all.

Mechanism, stated plainly. A metaphor is a coupling added to a body's precision
matrix J. As the coupling strength grows it dominates J, so the covariance, and
with it the relative phase on the coupled pair, approaches the coupling's target
regardless of the body's own baseline. Two bodies with different baseline J
therefore converge on the same relative phase under a shared metaphor. A
joint-angle instruction sets a mean pose and leaves J untouched, so each body
keeps its own baseline coordination and the two never converge.

This script has no new data. It runs on the model already fit to CMU mocap.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from taichi_model import (
    N, JOINTS, IDX, baseline_edges, covariance,
    participation_ratio, correlation_from_cov, relational_precision,
)

INK, ACCENT, COOL, MUTE = "#1b2430", "#b3541e", "#2f6b7a", "#9aa0a6"
TEAL, RUST = "#2f6b7a", "#b3541e"

# --------------------------------------------------------------------------- #
#  relational couplings -> precision.  relational_precision now lives in         #
#  taichi_model as the single source; a term is (i, j, a, b, w): 1/2 w(a x_i - b x_j)^2. #
# --------------------------------------------------------------------------- #

def cross_arm_terms(sign, w):
    """Couple the two arms limb-for-limb. sign=-1 mirror (anti-phase), +1 parallel."""
    pairs = [("L_shoulder", "R_shoulder"), ("L_elbow", "R_elbow"), ("L_wrist", "R_wrist")]
    return [(IDX[a], IDX[b], 1, sign, w) for a, b in pairs]

def within_arm_terms(w):
    out = []
    for s in ("L", "R"):
        j = [IDX[f"{s}_shoulder"], IDX[f"{s}_elbow"], IDX[f"{s}_wrist"]]
        out += [(j[0], j[1], 1, 1, w), (j[1], j[2], 1, 1, w)]
    return out

def arm_order_parameter(Sigma):
    """Relative-phase order parameter for the arm pair: correlation of a
    left-arm summary DOF with a right-arm summary DOF. +1 in-phase (phi=0),
    -1 anti-phase (phi=pi)."""
    L = [IDX["L_shoulder"], IDX["L_elbow"], IDX["L_wrist"]]
    R = [IDX["R_shoulder"], IDX["R_elbow"], IDX["R_wrist"]]
    uL = np.zeros(N); uL[L] = 1 / len(L)
    uR = np.zeros(N); uR[R] = 1 / len(R)
    vLL, vRR, vLR = uL @ Sigma @ uL, uR @ Sigma @ uR, uL @ Sigma @ uR
    return vLR / np.sqrt(vLL * vRR)


# --------------------------------------------------------------------------- #
#  Bodies. A body = its own baseline (gamma, base weight, idiosyncratic arm     #
#  coupling) plus a segment-length scale for the drawing.                       #
# --------------------------------------------------------------------------- #
class Body:
    def __init__(self, name, gamma, w_base, idio_sign, idio_w, seg_scale, color):
        self.name, self.gamma, self.w_base = name, gamma, w_base
        self.idio = cross_arm_terms(idio_sign, idio_w) + within_arm_terms(0.3)
        self.seg_scale, self.color = seg_scale, color

    def cov(self, metaphor_sign=None, beta=0.0):
        """Covariance with a metaphor (a cross-arm coupling) at strength beta."""
        be, bw = baseline_edges(self.w_base)
        terms = list(self.idio)
        if metaphor_sign is not None and beta > 0:
            terms += cross_arm_terms(metaphor_sign, beta) + within_arm_terms(0.6 * beta)
        return covariance(relational_precision(be, bw, terms, gamma=self.gamma))

# Teacher already trained: a strong mirror metaphor is baked into the body.
TEACHER = Body("teacher", gamma=0.50, w_base=0.15, idio_sign=-1, idio_w=1.2, seg_scale=1.00, color=INK)
# Two learners, different anatomy, both with in-phase (positive) baseline arms.
L1 = Body("learner A", gamma=0.60, w_base=0.24, idio_sign=+1, idio_w=1.0, seg_scale=1.35, color=TEAL)
L2 = Body("learner B", gamma=0.38, w_base=0.10, idio_sign=+1, idio_w=0.5, seg_scale=0.70, color=RUST)

MIRROR = -1  # the shared metaphor: cloud-hands mirror -> anti-phase arms


# --------------------------------------------------------------------------- #
#  Geometry for the skeleton drawing (inlined from run_demo v0)                 #
# --------------------------------------------------------------------------- #
PARENT = {"torso": None,
    "L_shoulder": "torso", "L_elbow": "L_shoulder", "L_wrist": "L_elbow",
    "R_shoulder": "torso", "R_elbow": "R_shoulder", "R_wrist": "R_elbow",
    "L_hip": "torso", "L_knee": "L_hip", "L_ankle": "L_knee",
    "R_hip": "torso", "R_knee": "R_hip", "R_ankle": "R_knee"}
NEUTRAL_DIR = {"torso": np.pi/2,
    "L_shoulder": np.deg2rad(150), "L_elbow": np.deg2rad(150), "L_wrist": np.deg2rad(150),
    "R_shoulder": np.deg2rad(30), "R_elbow": np.deg2rad(30), "R_wrist": np.deg2rad(30),
    "L_hip": np.deg2rad(255), "L_knee": np.deg2rad(270), "L_ankle": np.deg2rad(270),
    "R_hip": np.deg2rad(285), "R_knee": np.deg2rad(270), "R_ankle": np.deg2rad(270)}
BASE_SEG = {j: (0.9 if j == "torso" else 0.55) for j in JOINTS}
ROOT = np.array([0.0, -0.4]); ANGLE_SCALE = 0.6

def fk(x, seg_scale):
    pos = {"torso": ROOT}
    for j in JOINTS:
        if PARENT[j] is None:
            continue
        th = NEUTRAL_DIR[j] + ANGLE_SCALE * x[IDX[j]]
        L = BASE_SEG[j] * (seg_scale if j != "torso" else 1.0)
        pos[j] = pos[PARENT[j]] + L * np.array([np.cos(th), np.sin(th)])
    return pos

def draw(ax, x, seg_scale, color, alpha):
    pos = fk(x, seg_scale)
    for j in JOINTS:
        if PARENT[j] is None:
            continue
        p, c = pos[PARENT[j]], pos[j]
        ax.plot([p[0], c[0]], [p[1], c[1]], color=color, alpha=alpha, lw=1.7, solid_capstyle="round")


# =========================================================================== #
#  Experiment                                                                   #
# =========================================================================== #
def main():
    rT = arm_order_parameter(TEACHER.cov())            # teacher's trained relative phase
    print("=" * 72)
    print("STEP 1  TWO-BODY CONVERGENCE  (no new data; model already fit to CMU)")
    print("=" * 72)
    print(f"\nteacher arm relative-phase order parameter r_T = {rT:+.3f} "
          f"({'anti-phase' if rT < 0 else 'in-phase'})")
    print("\nbaseline (no metaphor), each learner keeps its own body:")
    for b in (L1, L2):
        print(f"   {b.name:9s}  r = {arm_order_parameter(b.cov()):+.3f}   "
              f"gap to teacher = {abs(arm_order_parameter(b.cov()) - rT):.3f}")

    betas = np.linspace(0, 4, 41)
    curves = {b.name: np.array([arm_order_parameter(b.cov(MIRROR, β)) for β in betas]) for b in (L1, L2)}

    # ---- convergence figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    ax.axhline(rT, color=INK, ls="--", lw=1.4, label=f"teacher target (r = {rT:+.2f})")
    ax.axhline(0, color=MUTE, lw=0.8)
    for b in (L1, L2):
        ax.plot(betas, curves[b.name], color=b.color, lw=2.4, label=b.name)
    ax.set_xlabel("metaphor strength  β  (cloud-hands mirror on the arms)")
    ax.set_ylabel("arm relative-phase order parameter r\n(+1 in-phase, −1 anti-phase)")
    ax.set_title("Two different bodies converge on the teacher's relative phase")
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    ax.set_ylim(-1.05, 1.05)

    ax = axes[1]
    gap = np.mean([np.abs(curves[b.name] - rT) for b in (L1, L2)], axis=0)
    base_gap = np.mean([abs(arm_order_parameter(b.cov()) - rT) for b in (L1, L2)])
    ax.plot(betas, gap, color=ACCENT, lw=2.6, label="metaphor channel")
    ax.axhline(base_gap, color=COOL, ls="--", lw=2.0,
               label="joint-angle channel (sets mean, not J)")
    ax.set_xlabel("metaphor strength  β")
    ax.set_ylabel("mean gap  |r_learner − r_teacher|")
    ax.set_title("A metaphor closes the gap; a joint-angle list cannot")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_ylim(0, max(gap.max(), base_gap) * 1.1)

    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Step 1. The metaphor is body-invariant, the joint-angle instruction is not",
                 fontsize=12, y=1.02, color=INK)
    fig.tight_layout()
    fig.savefig("fig_two_body.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("\nunder a strong metaphor (β = 4):")
    for b in (L1, L2):
        print(f"   {b.name:9s}  r = {curves[b.name][-1]:+.3f}   "
              f"gap to teacher = {abs(curves[b.name][-1] - rT):.3f}")
    print(f"\n   joint-angle channel gap stays {base_gap:.3f} at every β "
          f"(a mean pose does not touch coordination)")
    print("wrote fig_two_body.png")

    # ---- skeleton illustration: same angles differ across bodies; same metaphor shares coordination ----
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    # (left) one joint-angle target, rendered on three differently sized bodies
    x_target = np.zeros(N)
    x_target[[IDX["L_shoulder"], IDX["L_elbow"]]] = 0.9
    x_target[[IDX["R_shoulder"], IDX["R_elbow"]]] = 0.9
    for b in (TEACHER, L1, L2):
        draw(axes[0], x_target, b.seg_scale, b.color, 0.9)
    axes[0].set_title("One joint-angle target on three bodies\n"
                      "different segment lengths reach different places", fontsize=10)
    # (right) the mirror coordination the metaphor imposes, drawn on each body
    x_mirror = np.zeros(N)
    x_mirror[[IDX["L_shoulder"], IDX["L_elbow"], IDX["L_wrist"]]] = 0.9
    x_mirror[[IDX["R_shoulder"], IDX["R_elbow"], IDX["R_wrist"]]] = -0.9
    for b in (TEACHER, L1, L2):
        draw(axes[1], x_mirror, b.seg_scale, b.color, 0.9)
    axes[1].set_title("The mirror coordination the metaphor imposes\n"
                      "the same relative phase in every body, at its own scale", fontsize=10)
    for a in axes:
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    fig.savefig("fig_two_body_skeletons.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("wrote fig_two_body_skeletons.png")


if __name__ == "__main__":
    main()
