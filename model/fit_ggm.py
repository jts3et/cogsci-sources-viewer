"""
fit_ggm.py -- proof-of-concept: fit the Gaussian graphical model of taichi_model
to real CMU MoCap joint-angle data.

Tests, on real bodies, the three premises the taichi model rests on:
  (A) DIMENSIONALITY   -- is real joint-angle covariance low-PR (a few synergies)?
  (B) STRUCTURE        -- does the GGM precision matrix (partial correlations)
                          concentrate on ANATOMICALLY ADJACENT joints, i.e. does it
                          recover the kinematic tree the way the model's Laplacian
                          is built from it?
  (C) TASK-DEPENDENCE  -- do effective DOF and, crucially, the SIGN pattern of
                          limb coupling change across tasks (walk / run / jump)?
                          This is the real-data proxy for "different metaphor,
                          different relational content at (nearly) fixed collapse."
"""
from __future__ import annotations
import glob, os
import numpy as np

DATA = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
#  CMU ASF skeleton topology (bone tree) -- parsed from 01.asf :hierarchy       #
# --------------------------------------------------------------------------- #
HIER = {
    "root": ["lhipjoint", "rhipjoint", "lowerback"],
    "lhipjoint": ["lfemur"], "lfemur": ["ltibia"], "ltibia": ["lfoot"],
    "lfoot": ["ltoes"],
    "rhipjoint": ["rfemur"], "rfemur": ["rtibia"], "rtibia": ["rfoot"],
    "rfoot": ["rtoes"],
    "lowerback": ["upperback"], "upperback": ["thorax"],
    "thorax": ["lowerneck", "lclavicle", "rclavicle"],
    "lowerneck": ["upperneck"], "upperneck": ["head"],
    "lclavicle": ["lhumerus"], "lhumerus": ["lradius"], "lradius": ["lwrist"],
    "lwrist": ["lhand", "lthumb"], "lhand": ["lfingers"],
    "rclavicle": ["rhumerus"], "rhumerus": ["rradius"], "rradius": ["rwrist"],
    "rwrist": ["rhand", "rthumb"], "rhand": ["rfingers"],
}

def bone_edges():
    E = set()
    for p, cs in HIER.items():
        for c in cs:
            E.add((p, c))
    return E

def contracted_adjacency(keep):
    """Undirected adjacency among `keep` joints, contracting dropped connectors
    so a path through only-dropped nodes still counts as an anatomical link."""
    import itertools
    # full undirected graph
    adj = {}
    for a, b in bone_edges():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    nodes = set(adj)
    drop = [n for n in nodes if n not in keep]
    for d in drop:
        nb = list(adj.get(d, []))
        for x, y in itertools.combinations(nb, 2):     # bridge neighbours
            adj.setdefault(x, set()).add(y)
            adj.setdefault(y, set()).add(x)
        for x in nb:                                    # remove d
            adj[x].discard(d)
        adj.pop(d, None)
    return {n: {m for m in adj.get(n, []) if m in keep} for n in keep}


# --------------------------------------------------------------------------- #
#  AMC parsing                                                                  #
# --------------------------------------------------------------------------- #
def parse_amc(path):
    """Return (columns, X) where columns is a list of 'joint.k' labels and
    X is [frames x dof]. Root 6-DOF (global pos/orientation) is dropped."""
    with open(path) as f:
        lines = [ln.strip() for ln in f]
    i = 0
    while i < len(lines) and not lines[i].startswith(":DEGREES"):
        i += 1
    i += 1
    frames = []
    cur = None
    order = []            # joint order from first frame
    seen = set()
    def is_frameno(s):
        return s.isdigit()
    for ln in lines[i:]:
        if not ln:
            continue
        if is_frameno(ln):
            if cur is not None:
                frames.append(cur)
            cur = {}
            continue
        if cur is None:
            continue
        toks = ln.split()
        name = toks[0]
        vals = [float(v) for v in toks[1:]]
        cur[name] = vals
        if name not in seen and name != "root":
            seen.add(name); order.append((name, len(vals)))
    if cur:
        frames.append(cur)
    cols = [f"{n}.{k}" for (n, c) in order for k in range(c)]
    X = np.empty((len(frames), len(cols)))
    for r, fr in enumerate(frames):
        row = []
        for (n, c) in order:
            v = fr.get(n, [0.0] * c)
            row.extend(v if len(v) == c else [0.0] * c)
        X[r] = row
    return cols, X


def load_category(paths):
    """Pool trials, centering EACH trial by its own mean (removes between-trial
    pose offset, keeps within-trial covariation). Returns (cols, X_pooled)."""
    ref_cols = None
    chunks = []
    for p in paths:
        cols, X = parse_amc(p)
        if ref_cols is None:
            ref_cols = cols
        if cols != ref_cols:                 # align by intersection (same skeleton -> same)
            idx = [cols.index(c) for c in ref_cols if c in cols]
            X = X[:, idx]
        X = X - X.mean(axis=0, keepdims=True)
        chunks.append(X)
    return ref_cols, np.vstack(chunks)


# --------------------------------------------------------------------------- #
#  GGM estimators                                                               #
# --------------------------------------------------------------------------- #
def participation_ratio(eigs):
    lam = np.clip(eigs, 0, None)
    return (lam.sum() ** 2) / (np.square(lam).sum())

def pr_cov(X):
    C = np.cov(X, rowvar=False)
    return participation_ratio(np.linalg.eigvalsh(C)), C

def pr_corr(X):
    C = np.corrcoef(X, rowvar=False)
    return participation_ratio(np.linalg.eigvalsh(C)), C

def partial_correlation(C):
    """GGM edge estimate: partial correlations from precision = inv(cov)."""
    J = np.linalg.inv(C + 1e-8 * np.eye(len(C)))
    d = np.sqrt(np.diag(J))
    P = -J / np.outer(d, d)
    np.fill_diagonal(P, 1.0)
    return P, J

def joint_of(col):
    return col.split(".")[0]

def aggregate_to_joints(M, cols, reducer=np.max):
    """Collapse a dof x dof matrix to joint x joint by reducing |M| over dof pairs."""
    joints = []
    for c in cols:
        j = joint_of(c)
        if j not in joints:
            joints.append(j)
    idx = {j: [k for k, c in enumerate(cols) if joint_of(c) == j] for j in joints}
    A = np.zeros((len(joints), len(joints)))
    for i, ji in enumerate(joints):
        for k, jk in enumerate(joints):
            block = np.abs(M[np.ix_(idx[ji], idx[jk])])
            A[i, k] = reducer(block) if block.size else 0.0
    return joints, A


# --------------------------------------------------------------------------- #
#  Run                                                                          #
# --------------------------------------------------------------------------- #
# balanced: one subject, two trials per task -> fair PR comparison (no cross-subject
# pooling, which would inflate a task's dimensionality with between-body variance).
CATS = {
    "walk": sorted(glob.glob(os.path.join(DATA, "07_0*.amc"))),
    "run":  sorted(glob.glob(os.path.join(DATA, "09_0*.amc"))),
    "jump": sorted(glob.glob(os.path.join(DATA, "13_1*.amc"))),
}

# coordination pairs to watch the SIGN of (correlation), aggregated over dof:
PAIRS = [
    ("lfemur", "rfemur",  "legs L/R"),
    ("lhumerus", "rhumerus", "arms L/R"),
    ("lhumerus", "rfemur", "L-arm / R-leg (contralateral)"),
    ("lhumerus", "lfemur", "L-arm / L-leg (ipsilateral)"),
]

def signed_pair_corr(C, cols, ja, jb, axis=0):
    """Correlation of the MATCHED principal axis (.axis; 0 = rx flexion) of the two
    joints -- the physiologically interpretable 'same-motion' coordination, not a
    spurious off-axis maximum."""
    ta, tb = f"{ja}.{axis}", f"{jb}.{axis}"
    if ta in cols and tb in cols:
        return C[cols.index(ta), cols.index(tb)]
    # fall back to first available dof of each
    ia = next((k for k, c in enumerate(cols) if joint_of(c) == ja), None)
    ib = next((k for k, c in enumerate(cols) if joint_of(c) == jb), None)
    return C[ia, ib] if ia is not None and ib is not None else np.nan

def main():
    print("=" * 74)
    print("FITTING THE GGM TO REAL CMU MOCAP  (per-joint Euler angles, root dropped)")
    print("=" * 74)

    results = {}
    keep_ref = None
    for cat, paths in CATS.items():
        if not paths:
            print(f"[skip {cat}: no files]"); continue
        cols, X = load_category(paths)
        # drop near-constant columns (clavicles ~1e-15, etc.)
        sd = X.std(axis=0)
        good = sd > 1e-4
        cols = [c for c, g in zip(cols, good) if g]
        X = X[:, good]
        # standardize for correlation-based analyses
        prc, Ccov = pr_cov(X)
        prr, Ccorr = pr_corr(X)
        P, J = partial_correlation(Ccorr)
        keep_joints = sorted({joint_of(c) for c in cols})
        results[cat] = dict(cols=cols, X=X, Ccorr=Ccorr, P=P,
                            pr_cov=prc, pr_corr=prr, n=len(X),
                            p=X.shape[1], joints=keep_joints)
        print(f"\n[{cat}]  {len(paths)} trials, n={len(X)} frames, p={X.shape[1]} DOF")
        print(f"   PR(raw covariance)    = {prc:6.2f} / {X.shape[1]}   "
              f"({100*prc/X.shape[1]:.0f}% of dims)")
        print(f"   PR(correlation)       = {prr:6.2f} / {X.shape[1]}   "
              f"({100*prr/X.shape[1]:.0f}% of dims)")

    # ---- (B) anatomical recovery, using walk (richest) ----
    print("\n" + "-" * 74)
    print("(B) DOES THE PRECISION GRAPH RECOVER THE KINEMATIC TREE?")
    print("-" * 74)
    for cat in results:
        r = results[cat]
        joints, A = aggregate_to_joints(r["P"], r["cols"], reducer=np.max)
        adj = contracted_adjacency(set(joints))
        adj_vals, non_vals = [], []
        for i, ji in enumerate(joints):
            for k in range(i + 1, len(joints)):
                jk = joints[k]
                v = A[i, k]
                if jk in adj.get(ji, set()):
                    adj_vals.append(v)
                else:
                    non_vals.append(v)
        adj_vals, non_vals = np.array(adj_vals), np.array(non_vals)
        # rank-based separation (AUC): P(random adjacent edge > random non-adjacent)
        allv = np.concatenate([adj_vals, non_vals])
        ranks = allv.argsort().argsort().astype(float)
        auc = (ranks[:len(adj_vals)].mean() - (len(adj_vals) - 1) / 2) / len(non_vals)
        print(f"   [{cat}] |partial corr|: adjacent mean={adj_vals.mean():.3f} "
              f"vs non-adjacent mean={non_vals.mean():.3f}  |  "
              f"separation AUC={auc:.3f}  (0.5=chance, 1=perfect)")

    # ---- (C) task-dependent SIGN of limb coupling ----
    print("\n" + "-" * 74)
    print("(C) SAME BODY, DIFFERENT TASK: SIGN OF LIMB COORDINATION")
    print("    (real-data proxy for 'metaphor changes relational content')")
    print("-" * 74)
    header = "   pair".ljust(38) + "".join(f"{c:>10}" for c in results)
    print(header)
    for ja, jb, lab in PAIRS:
        row = f"   {lab}".ljust(38)
        for cat in results:
            r = results[cat]
            if ja in r["joints"] and jb in r["joints"]:
                v = signed_pair_corr(r["Ccorr"], r["cols"], ja, jb)
                row += f"{v:+10.2f}"
            else:
                row += f"{'--':>10}"
        print(row)
    print("\n(sign flips across columns = the coordination RELATION changes with task,")
    print(" at broadly similar dimensionality -- the sign-blind-PR claim, in real data.)")

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cats = list(results)
        fig, axes = plt.subplots(1, len(cats), figsize=(5.2 * len(cats), 4.6))
        if len(cats) == 1:
            axes = [axes]
        ANAT_ORDER = ["lowerback", "upperback", "thorax", "lowerneck", "upperneck",
                      "head", "rhumerus", "rradius", "rwrist", "rhand", "rthumb",
                      "rfingers", "lhumerus", "lradius", "lwrist", "lhand", "lthumb",
                      "lfingers", "rfemur", "rtibia", "rfoot", "rtoes",
                      "lfemur", "ltibia", "lfoot", "ltoes"]
        for ax, cat in zip(axes, cats):
            r = results[cat]
            # joint x joint from the MATCHED flexion axis (.0) -> shows true sign,
            # ordered anatomically so L/R limb blocks are contiguous
            joints = [j for j in ANAT_ORDER
                      if j in r["joints"] and f"{j}.0" in r["cols"]]
            ci = {j: r["cols"].index(f"{j}.0") for j in joints}
            A = np.array([[r["Ccorr"][ci[a], ci[b]] for b in joints] for a in joints])
            im = ax.imshow(A, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(joints))); ax.set_yticks(range(len(joints)))
            ax.set_xticklabels(joints, rotation=90, fontsize=6)
            ax.set_yticklabels(joints, fontsize=6)
            ax.set_title(f"{cat}   PR_corr={r['pr_corr']:.1f}/{r['p']}", fontsize=10)
        fig.colorbar(im, ax=axes, shrink=0.7, label="signed joint correlation")
        fig.suptitle("Real CMU MoCap: joint-correlation structure by task "
                     "(red=in-phase, blue=anti-phase)", fontsize=12)
        out = os.path.join(DATA, "fig_cmu_ggm.png")
        fig.savefig(out, bbox_inches="tight", dpi=130)
        print(f"\nwrote {out}")
    except Exception as e:
        print(f"\n[figure skipped: {e}]")


if __name__ == "__main__":
    main()
