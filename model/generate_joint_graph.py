"""
generate_joint_graph.py -- the body-as-graph diagram for companion.html.

Draws the 13 joints as dots and the kinematic tree as bone edges (from the same
geometry two_body.py uses), then adds one signed metaphor edge between the wrists:
a green "+" (in-phase) in the left panel, a clay "-" (anti-phase) in the right.
Writes fig_joint_graph.png. No data; pure model geometry.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from taichi_model import JOINTS, IDX, KINEMATIC_TREE
from two_body import fk, N

# companion.html palette
INK, BONE, JADE, CLAY, PAPER = "#16232E", "#2E6E8E", "#15A38A", "#C0603A", "#FFFFFF"

pos = fk(np.zeros(N), 1.0)   # neutral pose, unit segment scale


def panel(ax, sign):
    accent = JADE if sign > 0 else CLAY
    # bones
    for a, b in KINEMATIC_TREE:
        pa, pb = pos[a], pos[b]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=BONE, lw=3, zorder=1,
                solid_capstyle="round")
    # joints
    P = np.array([pos[j] for j in JOINTS])
    ax.scatter(P[:, 0], P[:, 1], s=90, color=INK, zorder=3, edgecolors=PAPER, linewidths=1.5)
    # the signed metaphor edge, wrist to wrist, as an arc over the body
    lw, rw = pos["L_wrist"], pos["R_wrist"]
    arc = FancyArrowPatch(lw, rw, connectionstyle="arc3,rad=-0.42", arrowstyle="-",
                          color=accent, lw=4, zorder=2, capstyle="round")
    ax.add_patch(arc)
    mid = (lw + rw) / 2 + np.array([0, 0.62])   # apex label
    ax.text(mid[0], mid[1], "+" if sign > 0 else "−", color="#fff", fontsize=20,
            fontweight="bold", ha="center", va="center", zorder=4,
            bbox=dict(boxstyle="circle,pad=0.28", fc=accent, ec="none"))
    ax.text(0, 1.5, "in-phase" if sign > 0 else "anti-phase", color=accent,
            fontsize=15, fontweight="bold", ha="center", va="center")
    ax.text(0, -2.45, "move together (0°)" if sign > 0 else "mirror (180°)",
            color="#5B6B72", fontsize=11.5, ha="center", va="center")
    ax.set_aspect("equal"); ax.set_xlim(-2.0, 2.0); ax.set_ylim(-2.7, 1.9)
    ax.axis("off")


fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.6))
panel(axes[0], +1)
panel(axes[1], -1)
fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.05)
fig.savefig("fig_joint_graph.png", dpi=130, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)
print("wrote fig_joint_graph.png")
