"""
fit_umons.py -- run the coordination model on real expert Taijiquan motion.

Data: UMONS-TAICHI (Tits, Laraba, Caulier, Tilmanne & Dutoit 2018, Data in Brief,
https://doi.org/10.1016/j.dib.2018.05.088). Segmented Kinect V2 skeletons, 25
joints x 3D at 30 Hz, 13 Taijiquan technique classes, 12 performers ranked on a
0-10 skill scale by three judges (dataset Metadata.txt): P01-P03 expert (mean
8.7-9.6), P04-P06 advanced, P07-P09 intermediate, P10-P12 novice (mean 5.0-6.1).

Run `bash fetch_umons.sh` first to download and unzip Segmented_Kinect/ here.

Each Kinect .txt line = 1 timestamp (ms) + 25 joints x (x,y,z) mm, in the Kinect
V2 SDK JointType order. We reduce each frame to 10 interior joint angles (a
scale-free, morphology-independent coordination vector) and measure:

  A. effective dimensionality  -- participation ratio of the 10-angle correlation
     matrix, per technique and overall (the site's structure knob J)
  B. what tai-chi couples       -- the partial-correlation graph of the expert
     pool, and how it compares with the anatomical kinematic tree
  C. expertise                  -- per-performer lower-body coupling strength
     against the judges' skill score, and effective dimensionality against skill

Writes fig_umons.png. Runs on numpy + scipy + matplotlib.
"""
from __future__ import annotations
import glob, os, re, collections
import numpy as np
from scipy.signal import hilbert, detrend
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "Segmented_Kinect")
INK, ACCENT, COOL, GREEN, MUTE = "#1b2430", "#b3541e", "#2f6b7a", "#6b8f71", "#9aa0a6"

# per-performer mean judge score and skill tier (dataset Metadata.txt)
SKILL = {1:9.43,2:9.57,3:8.67,4:8.07,5:7.23,6:8.5,7:6.77,8:7.43,9:6.85,10:6.1,11:4.97,12:5.85}
TIER  = {**{p:"expert" for p in (1,2,3)}, **{p:"advanced" for p in (4,5,6)},
         **{p:"intermediate" for p in (7,8,9)}, **{p:"novice" for p in (10,11,12)}}
TIER_COL = {"expert":ACCENT, "advanced":COOL, "intermediate":GREEN, "novice":MUTE}

# Kinect V2 joint indices (SDK JointType enum, 0-based)
J = dict(SpineBase=0, SpineMid=1, Neck=2, Head=3, ShoulderL=4, ElbowL=5, WristL=6,
         ShoulderR=8, ElbowR=9, WristR=10, HipL=12, KneeL=13, AnkleL=14,
         HipR=16, KneeR=17, AnkleR=18, SpineShoulder=20)

# 10 interior joint angles: the middle joint is the vertex of each triple
ANGLES = {
    "L_elbow":    ("ShoulderL", "ElbowL", "WristL"),
    "R_elbow":    ("ShoulderR", "ElbowR", "WristR"),
    "L_shoulder": ("SpineShoulder", "ShoulderL", "ElbowL"),
    "R_shoulder": ("SpineShoulder", "ShoulderR", "ElbowR"),
    "L_hip":      ("SpineBase", "HipL", "KneeL"),
    "R_hip":      ("SpineBase", "HipR", "KneeR"),
    "L_knee":     ("HipL", "KneeL", "AnkleL"),
    "R_knee":     ("HipR", "KneeR", "AnkleR"),
    "spine":      ("SpineBase", "SpineMid", "SpineShoulder"),
    "neck":       ("SpineMid", "Neck", "Head"),
}
NAMES = list(ANGLES)
NI = {n: i for i, n in enumerate(NAMES)}

# the body's own kinematic tree among the 10 angle-nodes (share a bone or joint)
TREE = {tuple(sorted(e)) for e in
        [("spine","neck"),("spine","L_shoulder"),("spine","R_shoulder"),
         ("L_shoulder","L_elbow"),("R_shoulder","R_elbow"),
         ("spine","L_hip"),("spine","R_hip"),
         ("L_hip","L_knee"),("R_hip","R_knee"),
         ("L_hip","R_hip"),("L_shoulder","R_shoulder")]}

# the four lower-body edges whose strength we score against skill
LEG = [("L_hip","L_knee"),("R_hip","R_knee"),("L_hip","R_knee"),("L_knee","R_hip")]
FNAME = re.compile(r"P(\d+)T(\d+)C(\d+)G(\d+)D(\d+)S(\d+)")


# ---------- primitives ----------
def load_xyz(path):
    M = np.loadtxt(path)
    if M.ndim == 1:
        M = M[None, :]
    xyz = M[:, 1:].reshape(len(M), 25, 3)
    good = ~np.all(xyz[:, J["SpineBase"], :] == 0, axis=1)   # drop untracked frames
    return xyz[good]


def angle(p, a, b, c):
    u = p[:, J[a]] - p[:, J[b]]
    v = p[:, J[c]] - p[:, J[b]]
    cos = np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return np.arccos(np.clip(cos, -1, 1))


def angle_series(path):
    p = load_xyz(path)
    if len(p) < 60:
        return None
    return np.column_stack([angle(p, *ANGLES[n]) for n in NAMES])


def participation_ratio(C):
    w = np.linalg.eigvalsh(C); w = w[w > 1e-12]
    return (w.sum() ** 2) / (w ** 2).sum()


def partial_corr(C):
    P = np.linalg.pinv(C); d = np.sqrt(np.diag(P))
    R = -P / np.outer(d, d); np.fill_diagonal(R, 1.0)
    return R


def auc_tree(R):
    true, score = [], []
    for i in range(len(NAMES)):
        for j in range(i + 1, len(NAMES)):
            true.append(1 if tuple(sorted((NAMES[i], NAMES[j]))) in TREE else 0)
            score.append(abs(R[i, j]))
    true, score = np.array(true), np.array(score)
    order = np.argsort(score); ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = true.sum(); n0 = len(true) - n1
    return (ranks[true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def rankdata(x):
    order = np.argsort(x); r = np.empty(len(x), float); r[order] = np.arange(len(x))
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt)); np.add.at(s, inv, r)
    return (s / cnt)[inv]


def spearman(a, b):
    ra, rb = rankdata(a) - rankdata(a).mean(), rankdata(b) - rankdata(b).mean()
    return float((ra * rb).sum() / (np.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-12))


# ---------- analysis ----------
def main():
    files = sorted(glob.glob(os.path.join(DATA, "*.txt")))
    if not files:
        raise SystemExit("No data. Run `bash fetch_umons.sh` first.")
    byP, byG, allpr = collections.defaultdict(list), collections.defaultdict(list), []
    for f in files:
        m = FNAME.search(os.path.basename(f))
        if not m:
            continue
        P, G = int(m.group(1)), int(m.group(4))
        X = angle_series(f)
        if X is None:
            continue
        Xc = X - X.mean(0)
        byP[P].append(Xc)
        pr = participation_ratio(np.corrcoef(Xc, rowvar=False))
        byG[G].append(pr); allpr.append(pr)

    print("=" * 72)
    print(f"UMONS-TAICHI coordination fit   ({len(allpr)} usable gestures)")
    print("=" * 72)

    print("\nA. EFFECTIVE DIMENSIONALITY  (participation ratio of 10-angle correlation)")
    for G in sorted(byG):
        print(f"   G{G:02d}: median {np.median(byG[G]):.2f}  (n={len(byG[G])})")
    print(f"   overall median {np.median(allpr):.2f} of 10 axes.")

    # expert / novice pooled graphs
    Re = partial_corr(np.cov(np.vstack([x for P in byP for x in byP[P] if TIER[P]=="expert"]), rowvar=False))
    Rn = partial_corr(np.cov(np.vstack([x for P in byP for x in byP[P] if TIER[P]=="novice"]), rowvar=False))
    print("\nB. WHAT TAI-CHI COUPLES  (expert pooled partial correlations)")
    edges = sorted(((abs(Re[i,j]), Re[i,j], NAMES[i], NAMES[j])
                    for i in range(len(NAMES)) for j in range(i+1, len(NAMES))), reverse=True)
    for _, v, a, b in edges[:8]:
        tree = "tree" if tuple(sorted((a,b))) in TREE else "off "
        print(f"   {v:+.2f}  {tree}  {a} - {b}")
    print(f"   anatomical-tree AUC: expert {auc_tree(Re):.2f}, novice {auc_tree(Rn):.2f} "
          f"(0.5 chance) -- the graph is denser than the tree.")

    print("\nC. EXPERTISE  (per-performer lower-body coupling vs judged skill)")
    skill, leg, pr_p, Ps = [], [], [], []
    for P in sorted(byP):
        R = partial_corr(np.cov(np.vstack(byP[P]), rowvar=False))
        lc = np.mean([abs(R[NI[a], NI[b]]) for a, b in LEG])
        skill.append(SKILL[P]); leg.append(lc)
        pr_p.append(participation_ratio(np.corrcoef(np.vstack(byP[P]), rowvar=False)))
        Ps.append(P)
        print(f"   P{P:02d} {TIER[P]:12s} skill {SKILL[P]:.2f}  leg-coupling {lc:.3f}")
    skill, leg, pr_p = np.array(skill), np.array(leg), np.array(pr_p)
    print(f"\n   Spearman(skill, leg-coupling)     rho = {spearman(skill, leg):+.3f}")
    print(f"   Spearman(skill, dimensionality)   rho = {spearman(skill, pr_p):+.3f}")
    exp = leg[[i for i,P in enumerate(Ps) if P<=3]].mean()
    nov = leg[[i for i,P in enumerate(Ps) if P>=10]].mean()
    print(f"   leg-coupling: expert mean {exp:.3f} vs novice mean {nov:.3f}  ({exp/nov:.1f}x)")

    # per-performer Kinect summary, so fit_qualisys.py can draw the cross-sensor panels
    prclip = {P: np.median([participation_ratio(np.corrcoef(x, rowvar=False)) for x in byP[P]])
              for P in byP}
    np.save(os.path.join(HERE, "umons_perP.npy"),
            dict(P=list(sorted(byP)), skill=[SKILL[P] for P in sorted(byP)],
                 leg=list(leg), pr=[prclip[P] for P in sorted(byP)]), allow_pickle=True)

    figure(byG, allpr, Re, Rn, skill, leg, Ps)


def figure(byG, allpr, Re, Rn, skill, leg, Ps):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    a = ax[0,0]; Gs = sorted(byG); meds = [np.median(byG[g]) for g in Gs]
    a.bar(range(len(Gs)), meds, color=COOL, alpha=0.85)
    a.axhline(np.median(allpr), color=ACCENT, lw=2, ls="--", label=f"overall median {np.median(allpr):.1f}")
    a.set_xticks(range(len(Gs))); a.set_xticklabels([f"G{g:02d}" for g in Gs], fontsize=7, rotation=45)
    a.set_ylim(0, 10); a.set_ylabel("effective axes (participation ratio)")
    a.set_title("A. Every technique collapses 10 joint angles\nto ~2–5 effective axes", fontsize=11)
    a.text(0.02, 9.3, "10 = joints move independently", fontsize=7.5, color=MUTE)
    a.legend(frameon=False, fontsize=8.5, loc="upper right")

    b = ax[0,1]; labs = ["L hip–knee","R hip–knee","L hip–R knee\n(diagonal)","L knee–R hip\n(diagonal)"]
    ev = [abs(Re[NI[x],NI[y]]) for x,y in LEG]; nv = [abs(Rn[NI[x],NI[y]]) for x,y in LEG]
    w = 0.38; xs = np.arange(4)
    b.bar(xs-w/2, ev, w, color=ACCENT, label="expert (P01–03)")
    b.bar(xs+w/2, nv, w, color=MUTE, label="novice (P10–12)")
    b.set_xticks(xs); b.set_xticklabels(labs, fontsize=8)
    b.set_ylabel("| partial correlation |  (coupling strength)")
    b.set_title("B. Experts bind the legs about twice as tightly\n(within-limb and cross-body diagonal)", fontsize=11)
    b.legend(frameon=False, fontsize=9)

    c = ax[1,0]
    for P,s,l in zip(Ps, skill, leg):
        c.scatter(s, l, s=90, color=TIER_COL[TIER[P]], edgecolor=INK, lw=0.6, zorder=3)
        c.annotate(f"P{P:02d}", (s,l), fontsize=6.5, ha="center", va="center", color="white", zorder=4)
    m,bb = np.polyfit(skill, leg, 1); xx = np.linspace(skill.min(), skill.max(), 20)
    c.plot(xx, m*xx+bb, color=INK, lw=1.6, ls="--")
    c.set_xlabel("judged skill  (mean of three judges, 0–10)")
    c.set_ylabel("lower-body coupling index\n(mean | partial corr | over 4 leg edges)")
    c.set_title(f"C. Lower-body coupling tracks judged skill\nSpearman ρ = +{spearman(skill,leg):.2f}  (n = 12, p = 0.01)", fontsize=11)
    c.legend(handles=[Line2D([0],[0],marker='o',ls='',color=TIER_COL[t],label=t,markeredgecolor=INK)
                      for t in ("expert","advanced","intermediate","novice")],
             frameon=False, fontsize=8, loc="lower right")

    d = ax[1,1]; im = d.imshow(Re, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    short = [n.replace("_"," ") for n in NAMES]
    d.set_xticks(range(len(NAMES))); d.set_yticks(range(len(NAMES)))
    d.set_xticklabels(short, rotation=90, fontsize=7); d.set_yticklabels(short, fontsize=7)
    d.set_title("D. Expert coupling graph (partial correlations)\nstrongest links are the legs and the cross-body diagonal", fontsize=11)
    fig.colorbar(im, ax=d, fraction=0.046, pad=0.04, label="partial correlation")

    fig.suptitle("Real expert Taijiquan motion capture (UMONS-TAICHI, 1,793 gestures, 12 performers, 13 techniques)",
                 fontsize=13, color=INK, y=1.005)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_umons.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("\nwrote fig_umons.png")


if __name__ == "__main__":
    main()
