"""
slide_figures.py -- the three order-parameter figures for companion.html.

  fig_order_parameter.png : what the order parameter is (relative phase + HKB well)
  fig_read_motion.png     : how it's read from motion (Hilbert pipeline)
  fig_skill_scatter.png   : the result (arm phase-locking tracks judged skill)

The first two are schematics (no data). The scatter is generated from the real
UMONS-TAICHI Qualisys markers, reusing fit_qualisys's own arm-PLV computation;
the 12 per-performer points are cached to qualisys_perP.npy on first run.
"""
from __future__ import annotations
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
INK, PAPER, JADE, RIVER, CLAY, MUTE = "#16232E", "#F7F9F8", "#15A38A", "#2E6E8E", "#C0603A", "#5B6B72"
LINE = "#DCE3E0"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Cambria", "Georgia", "DejaVu Serif"],
                     "svg.fonttype": "none"})


# --------------------------------------------------------------------------- #
def fig_order_parameter():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})

    # -- left: two limbs, a phase offset phi --
    t = np.linspace(0, 4 * np.pi, 400)
    axL.plot(t, np.cos(t) + 3.0, color=RIVER, lw=2.6)
    axL.plot(t, np.cos(t - 1.1) + 0.2, color=JADE, lw=2.6)
    axL.text(0.1, 4.15, "limb 1", color=RIVER, fontsize=11, style="italic")
    axL.text(0.1, 1.35, "limb 2", color=JADE, fontsize=11, style="italic")
    p1, p2 = 0.0, 1.1                      # peak of limb1 at t=0, limb2 at t=1.1
    axL.annotate("", xy=(p2, 4.02), xytext=(p1, 4.02),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
    axL.text((p1 + p2) / 2, 4.5, r"$\varphi$", color=INK, fontsize=16, ha="center")
    axL.set_title("What it measures", color=RIVER, fontsize=13, fontweight="bold", loc="left")
    axL.text(0.5, -0.14, "one number for how the two are timed", transform=axL.transAxes,
             ha="center", color=MUTE, fontsize=10.5, style="italic")
    axL.set_xlim(-0.3, 4 * np.pi); axL.set_ylim(-1.2, 5.1); axL.axis("off")

    # -- right: HKB double-well potential over relative phase --
    phi = np.linspace(0, 2 * np.pi, 500)
    V = -np.cos(phi) - 0.55 * np.cos(2 * phi)          # wells at 0/2pi (deep) and pi (shallow)
    axR.plot(np.degrees(phi), V, color=INK, lw=3, solid_capstyle="round")
    # a ball resting in the in-phase well at 0 deg
    axR.plot(6, V[0] + 0.02, "o", color=CLAY, ms=15, zorder=5)
    axR.annotate("a ridge between\nsettled patterns", xy=(90, -np.cos(np.pi/2) - 0.55*np.cos(np.pi)),
                 xytext=(120, 1.15), color=MUTE, fontsize=9.5, ha="left",
                 arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.2))
    axR.text(0, V[0] - 0.42, "in-phase\nlimbs together", color=RIVER, fontsize=11,
             fontweight="bold", ha="left", va="top")
    axR.text(360, V[0] - 0.42, "in-phase", color=RIVER, fontsize=11, fontweight="bold",
             ha="right", va="top")
    axR.text(180, V[len(phi)//2] - 0.3, "anti-phase\nlimbs mirror", color=CLAY, fontsize=11,
             fontweight="bold", ha="center", va="top")
    axR.set_title("Where it settles", color=JADE, fontsize=13, fontweight="bold", loc="left")
    axR.set_xlabel(r"relative phase  $\varphi$", color=INK, fontsize=11)
    axR.set_xticks([0, 180, 360]); axR.set_xticklabels(["0°", "180°", "360°"], color=MUTE)
    axR.set_yticks([]); axR.set_ylim(-2.0, 1.5)
    for s in ("top", "right", "left"):
        axR.spines[s].set_visible(False)
    axR.spines["bottom"].set_color(LINE)

    fig.suptitle("The order parameter is where movement settles", fontsize=15, fontweight="bold",
                 color=INK, x=0.02, ha="left", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(HERE, "fig_order_parameter.png"), dpi=130, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig); print("wrote fig_order_parameter.png")


# --------------------------------------------------------------------------- #
def fig_read_motion():
    fig, ax = plt.subplots(figsize=(9.8, 3.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(1.5, 97, "How the order parameter is read from motion",
            fontsize=14.5, fontweight="bold", color=INK, va="top", ha="left")
    steps = [
        ("DATA", r"$x_1(t),\ x_2(t)$", "limb flexion angles", "recorded by motion capture", RIVER),
        ("PHASE", r"$\theta_i=\mathrm{arg}[x_i+i\,\mathcal{H}\{x_i\}]$", "each limb's phase", "detrended, then Hilbert", RIVER),
        ("RELATIVE PHASE", r"$\varphi(t)=\theta_1(t)-\theta_2(t)$", "how the two are timed", "wrapped, instant by instant", JADE),
        ("ORDER PARAMETER", r"$\bar\varphi=\mathrm{arg}\,\langle e^{i\varphi}\rangle$", "a direction and a tightness", "averaged over the movement", JADE),
    ]
    w, gap = 22.2, 2.6; x0 = 1.5; ytop, h = 52, 30
    for k, (head, eq, cap1, cap2, col) in enumerate(steps):
        x = x0 + k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, ytop), w, h, boxstyle="round,pad=0.5,rounding_size=2",
                                    fc="#FBFCFC", ec=LINE, lw=1.3))
        ax.text(x + w / 2, ytop + h - 4.5, head, color=col, fontsize=9.8, fontweight="bold", ha="center")
        ax.text(x + w / 2, ytop + h / 2 + 0.5, eq, color=INK, fontsize=10.6, ha="center", va="center")
        ax.text(x + w / 2, ytop + 5.4, cap1, color=INK, fontsize=8.3, ha="center")
        ax.text(x + w / 2, ytop + 2.2, cap2, color=MUTE, fontsize=7.4, ha="center", style="italic")
        if k < 3:
            ax.add_patch(FancyArrowPatch((x + w + 0.1, ytop + h / 2), (x + w + gap - 0.1, ytop + h / 2),
                                         arrowstyle="-|>", mutation_scale=12, color=MUTE, lw=1.5))
    ax.plot([6, 94], [43, 43], color=LINE, lw=1)     # divider
    ax.text(50, 35, r"$\mathrm{PLV}=|\langle e^{i\varphi}\rangle|$  —  which pattern $\bar\varphi$, and how steady",
            ha="center", fontsize=11.5, color=INK)
    for cx, lab0, lab1, c0, c1 in [(27, "0°  in-phase", "180°  anti-phase", RIVER, CLAY),
                                   (73, "0  drifting", "1  tightly locked", MUTE, JADE)]:
        ax.plot([cx - 16, cx + 16], [23, 23], color=LINE, lw=2.5, solid_capstyle="round")
        ax.plot(cx - 16, 23, "o", color=c0, ms=8); ax.plot(cx + 16, 23, "o", color=c1, ms=8)
        ax.text(cx - 16, 17, lab0, color=c0, fontsize=8.2, ha="center", fontweight="bold")
        ax.text(cx + 16, 17, lab1, color=c1, fontsize=8.2, ha="center", fontweight="bold")
    ax.text(50, 6, "on the tai-chi markers, the arms' phase-locking rises with judged skill  ·  "
                   r"Spearman $\rho=+0.68$", ha="center", fontsize=9, color=MUTE, style="italic")
    fig.savefig(os.path.join(HERE, "fig_read_motion.png"), dpi=130, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig); print("wrote fig_read_motion.png")


# --------------------------------------------------------------------------- #
def _qualisys_points():
    """(P, skill, arm-PLV) per performer, cached; computed from the real markers on first run."""
    cache = os.path.join(HERE, "qualisys_perP.npy")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True).item()
        return d["P"], d["skill"], d["arm"], d["tier"]
    import fit_qualisys as fq
    cap = {}
    fq.figure = lambda Ps, sk, armv, legc, prv, Re: cap.update(P=Ps, sk=sk, arm=armv)
    fq.main()
    P = list(cap["P"]); d = {"P": P, "skill": [fq.SKILL[p] for p in P],
                             "arm": list(cap["arm"]), "tier": [fq.TIER[p] for p in P]}
    np.save(cache, d); return d["P"], d["skill"], d["arm"], d["tier"]


def fig_skill_scatter():
    P, skill, arm, tier = _qualisys_points()
    skill, arm = np.array(skill, float), np.array(arm, float)
    col = {"expert": CLAY, "advanced": RIVER, "intermediate": JADE, "novice": "#9AA0A6"}
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for p, s, a, tr in zip(P, skill, arm, tier):
        ax.scatter(s, a, s=190, color=col[tr], edgecolor=INK, lw=0.7, zorder=3)
        ax.annotate(f"P{p:02d}", (s, a), fontsize=6.5, ha="center", va="center", color="white", zorder=4)
    m, b = np.polyfit(skill, arm, 1); xx = np.linspace(skill.min() - 0.3, skill.max() + 0.3, 20)
    ax.plot(xx, m * xx + b, "--", color=INK, lw=1.7, zorder=2)
    ax.set_xlabel("judged skill  (three judges, 0–10)", fontsize=11.5, color=INK)
    ax.set_ylabel("arm phase-locking value\n(bilateral, 179 Hz)", fontsize=11.5, color=INK)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc=col[t], mec=INK, label=t)
                       for t in ("expert", "advanced", "intermediate", "novice")],
              frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Skill shows up as a steadier order parameter", fontsize=14, fontweight="bold",
                 color=INK, loc="left", pad=12)
    ax.text(0, -0.16, r"Spearman $\rho = +0.68$  (n = 12, p = 0.015).  The amount of coupling does not track skill.",
            transform=ax.transAxes, fontsize=10, color=MUTE, style="italic")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTE)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_skill_scatter.png"), dpi=130, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig); print("wrote fig_skill_scatter.png")


if __name__ == "__main__":
    fig_order_parameter()
    fig_read_motion()
    fig_skill_scatter()
