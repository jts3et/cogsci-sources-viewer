"""
taichi_model.py -- core Gaussian-graphical-model of body coordination (v0).
"""

from __future__ import annotations
import numpy as np


JOINTS = [
    "torso",
    "L_shoulder", "L_elbow", "L_wrist",
    "R_shoulder", "R_elbow", "R_wrist",
    "L_hip", "L_knee", "L_ankle",
    "R_hip", "R_knee", "R_ankle",
]
IDX = {name: i for i, name in enumerate(JOINTS)}
N = len(JOINTS)

KINEMATIC_TREE = [
    ("torso", "L_shoulder"), ("L_shoulder", "L_elbow"), ("L_elbow", "L_wrist"),
    ("torso", "R_shoulder"), ("R_shoulder", "R_elbow"), ("R_elbow", "R_wrist"),
    ("torso", "L_hip"), ("L_hip", "L_knee"), ("L_knee", "L_ankle"),
    ("torso", "R_hip"), ("R_hip", "R_knee"), ("R_knee", "R_ankle"),
]


def clique(joint_names):
    js = [IDX[j] for j in joint_names]
    return [(a, b) for i, a in enumerate(js) for b in js[i + 1:]]


METAPHORS = {
    "pearls_on_a_string": clique(["R_shoulder", "R_elbow", "R_wrist"]),
    "cloud_hands": clique(["L_wrist", "L_elbow", "L_shoulder",
                           "torso",
                           "R_shoulder", "R_elbow", "R_wrist"]),
    "be_the_mountain": clique(["torso", "L_hip", "R_hip",
                               "L_knee", "R_knee", "L_ankle", "R_ankle"]),
}


def signed_laplacian(edges, n=N):
    """The one precision object. edges: iterable of (i, j, w) with SIGNED w.

    Absolute-degree (|W|) diagonal, D̄ = diag(Σⱼ|Wᵢⱼ|), keeps J = γI + (D̄ − W)
    positive-definite even when an edge is negative. A positive edge gives the
    ordinary Laplacian block (in-phase); a negative edge the signless block D̄ + |W|
    (anti-phase). Anatomy and metaphor are the same object on disjoint supports.
    """
    W = np.zeros((n, n))
    for (i, j, w) in edges:
        W[i, j] += w
        W[j, i] += w
    Dbar = np.diag(np.abs(W).sum(axis=1))
    return Dbar - W


def precision_signed(edges, gamma=0.5, n=N):
    return gamma * np.eye(n) + signed_laplacian(edges, n)


def laplacian(edges, weights, n=N):
    """Back-compatible wrapper: a Laplacian from parallel (edge, weight) lists.
    Now built through the signed builder; identical for non-negative weights."""
    return signed_laplacian([(a, b, w) for (a, b), w in zip(edges, weights)], n)


def precision_matrix(base_edges, base_w, metaphor_edges=None, beta=0.0, gamma=0.5,
                     metaphor_sign=+1):
    """Single precision builder: J = γI + signed_laplacian(anatomy[+] ∪ β·metaphor[signed]).

    metaphor_sign = +1 in-phase (a clique that agrees), −1 anti-phase (mirror).
    For the default in-phase case this reproduces the old clique-Laplacian exactly.
    """
    edges = [(a, b, w) for (a, b), w in zip(base_edges, base_w)]
    if metaphor_edges is not None and beta > 0:
        edges += [(a, b, metaphor_sign * beta) for (a, b) in metaphor_edges]
    return precision_signed(edges, gamma=gamma)


def relational_precision(base_edges, base_w, terms=None, gamma=0.5, n=N):
    """General signed-coupling precision, the superset precision_signed specializes.

    A relational term (i, j, a, b, w) is the perfect-square penalty 1/2 w (a x_i - b x_j)^2,
    adding w*a^2 and w*b^2 to the diagonal and -w*a*b off-diagonal. For |a| = |b| = 1 on
    distinct pairs this equals precision_signed on scalar signed edges (a positive edge is
    a = b, an anti-phase edge a = -b). Two forms it carries that a scalar signed_laplacian
    cannot: a fixed ratio (a = 1, b = r) at gain r, and two couplings on the SAME pair,
    whose absolute weights accumulate on the diagonal term by term (the frustration case)
    instead of cancelling first. That per-term accumulation is the single source for
    two_body.py and frustration.py.
    """
    J = gamma * np.eye(n) + laplacian(base_edges, base_w, n)
    for (i, j, a, b, w) in (terms or []):
        J[i, i] += w * a * a
        J[j, j] += w * b * b
        J[i, j] -= w * a * b
        J[j, i] -= w * a * b
    return J


def covariance(J, T=1.0):
    return T * np.linalg.inv(J)


def participation_ratio(Sigma):
    lam = np.linalg.eigvalsh(Sigma)
    lam = np.clip(lam, 0, None)
    return (lam.sum() ** 2) / (np.square(lam).sum())


def correlation_from_cov(Sigma):
    d = np.sqrt(np.diag(Sigma))
    return Sigma / np.outer(d, d)


def baseline_edges(w_base=0.15):
    edges = [(IDX[a], IDX[b]) for a, b in KINEMATIC_TREE]
    weights = [w_base] * len(edges)
    return edges, weights


if __name__ == "__main__":
    be, bw = baseline_edges()
    J0 = precision_matrix(be, bw)
    pr0 = participation_ratio(covariance(J0, T=1.0))

    Jm = precision_matrix(be, bw, METAPHORS["pearls_on_a_string"], beta=3.0)
    pr_m = participation_ratio(covariance(Jm, T=1.0))
    pr_m_hotT = participation_ratio(covariance(Jm, T=5.0))

    print(f"baseline PR                 : {pr0:6.3f}  (of {N} DOF)")
    print(f"+ metaphor (T=1)  PR        : {pr_m:6.3f}   <- DOF collapse")
    print(f"+ metaphor (T=5)  PR        : {pr_m_hotT:6.3f}   <- identical: T doesn't touch PR")
