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


def laplacian(edges, weights, n=N):
    W = np.zeros((n, n))
    for (a, b), w in zip(edges, weights):
        W[a, b] += w
        W[b, a] += w
    D = np.diag(W.sum(axis=1))
    return D - W


def precision_matrix(base_edges, base_w, metaphor_edges=None, beta=0.0, gamma=0.5):
    L = laplacian(base_edges, base_w)
    if metaphor_edges is not None and beta > 0:
        Lm = laplacian(metaphor_edges, [1.0] * len(metaphor_edges))
        L = L + beta * Lm
    J = gamma * np.eye(N) + L
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
