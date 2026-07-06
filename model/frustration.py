"""
frustration.py -- STEP 3 of the intended outcome: frustration as curriculum.

Claim to test (Methodology, section 3 and section 6): two conflicting teaching
images loaded on the same joints cannot both be satisfied, so the coordinated
mode freezes and the coordination collapses through zero. No new data.

We load an in-phase coupling at weight W(1-lambda) and an anti-phase coupling at
weight W*lambda on the same arm pair, and sweep lambda from 0 (pure in-phase) to
1 (pure anti-phase). We track the variance of the two collective coordinates,
the together-mode c+ and the opposed-mode c-, and the arm-arm correlation.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from taichi_model import N, IDX, baseline_edges, laplacian, covariance

INK, ACCENT, COOL, MUTE = "#1b2430", "#b3541e", "#2f6b7a", "#9aa0a6"


def relational_precision(be, bw, terms, gamma=0.5):
    J = gamma * np.eye(N) + laplacian(be, bw)
    for (i, j, a, b, w) in terms:
        J[i, i] += w * a * a
        J[j, j] += w * b * b
        J[i, j] -= w * a * b
        J[j, i] -= w * a * b
    return J


PAIRS = [("L_shoulder", "R_shoulder"), ("L_elbow", "R_elbow"), ("L_wrist", "R_wrist")]
Larm = [IDX["L_shoulder"], IDX["L_elbow"], IDX["L_wrist"]]
Rarm = [IDX["R_shoulder"], IDX["R_elbow"], IDX["R_wrist"]]

def arm_corr(S):
    uL = np.zeros(N); uL[Larm] = 1 / 3
    uR = np.zeros(N); uR[Rarm] = 1 / 3
    return (uL @ S @ uR) / np.sqrt((uL @ S @ uL) * (uR @ S @ uR))


def main():
    be, bw = baseline_edges()
    W = 3.0
    lams = np.linspace(0, 1, 61)

    # collective coordinates of the arm pair: c+ (together / sum), c- (opposed / difference)
    uplus = np.zeros(N); uplus[Larm] = 1; uplus[Rarm] = 1; uplus /= np.linalg.norm(uplus)
    uminus = np.zeros(N); uminus[Larm] = 1; uminus[Rarm] = -1; uminus /= np.linalg.norm(uminus)

    vp, vm, corr = [], [], []
    for lam in lams:
        terms = []
        for a_, b_ in PAIRS:
            terms.append((IDX[a_], IDX[b_], 1, 1, W * (1 - lam)))   # in-phase, penalizes difference
            terms.append((IDX[a_], IDX[b_], 1, -1, W * lam))        # anti-phase, penalizes sum
        S = covariance(relational_precision(be, bw, terms))
        vp.append(uplus @ S @ uplus)
        vm.append(uminus @ S @ uminus)
        corr.append(arm_corr(S))
    vp, vm, corr = np.array(vp), np.array(vm), np.array(corr)
    free = np.maximum(vp, vm)   # the cheaper (higher-variance) mode still available

    print("=" * 70)
    print("STEP 3  FRUSTRATION AS CURRICULUM  (no new data)")
    print("=" * 70)
    print(f"\n   lambda=0 (one image, in-phase):  free mode variance = {free[0]:.2f}, arm r = {corr[0]:+.2f}")
    i_half = np.argmin(np.abs(lams - 0.5))
    print(f"   lambda=0.5 (two conflicting):     free mode variance = {free[i_half]:.2f}, arm r = {corr[i_half]:+.2f}")
    print(f"   lambda=1 (one image, anti-phase): free mode variance = {free[-1]:.2f}, arm r = {corr[-1]:+.2f}")
    print(f"\n   the cheapest coordinated mode is smallest at lambda = {lams[np.argmin(free)]:.2f} "
          f"(variance {free.min():.2f}), the frozen point")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot(lams, vp, color=COOL, lw=2.4, label="together-mode c+ (both arms same way)")
    ax.plot(lams, vm, color=ACCENT, lw=2.4, label="opposed-mode c− (arms mirror)")
    ax.axvline(0.5, color=MUTE, ls="--", lw=1.2)
    ax.set_xlabel("conflict  λ   (0 = one image, in-phase   →   1 = one image, anti-phase)")
    ax.set_ylabel("variance of the coordinated mode\n(high = free to move, low = frozen)")
    ax.set_title("At λ = ½ both coordinated modes are suppressed")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ax.plot(lams, corr, color=INK, lw=2.6)
    ax.axhline(0, color=MUTE, lw=0.8)
    ax.axvline(0.5, color=MUTE, ls="--", lw=1.2)
    ax.set_xlabel("conflict  λ")
    ax.set_ylabel("arm–arm correlation")
    ax.set_title("Two images do not average, they cancel:\nthe coordination passes through zero at λ = ½")
    ax.set_ylim(-1.05, 1.05)

    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Step 3. Conflicting teaching images freeze the coordination they share",
                 fontsize=12, y=1.02, color=INK)
    fig.tight_layout()
    fig.savefig("fig_frustration.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("wrote fig_frustration.png")


if __name__ == "__main__":
    main()
