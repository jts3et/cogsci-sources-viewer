"""
recover_decompose.py -- the model as one operation: mocap -> J -> (anatomy + metaphor).

    cov (empirical)  ->  J_hat = inv(cov)                          [recover]
    J_hat            ->  L_anatomy (on the kinematic tree)
                         + M (everything off the tree, SIGNED)     [decompose]
    M's limb-to-limb entries ARE the task/metaphor coordination.
    sign of a J off-diagonal:  < 0 = in-phase link,  > 0 = anti-phase link.

One object throughout: a signed graph Laplacian (signed_laplacian.py). The forward
model builds it; this recovers it from data and reads the metaphor off the remainder.
Anatomy is not fit or assumed away -- it is exactly the on-tree partial couplings;
the metaphor is exactly the off-tree ones. Decomposition by graph support.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signed_laplacian import signed_laplacian, precision_signed, tree_edges, cross_arm_edges, within_arm_edges
from taichi_model import N, IDX, JOINTS, KINEMATIC_TREE, covariance

def recover_precision(cov, ridge=1e-9):
    n = len(cov)
    return np.linalg.inv(cov + ridge*np.eye(n))

def decompose(J_hat, tree_idx_edges):
    """J_hat -> (L_anatomy, M, w_anat). Anatomy weight on a tree edge is the negative
    of the empirical precision off-diagonal there; M is the remainder (its off-diagonal
    is ~0 on tree edges by construction, nonzero only off-tree = the metaphor)."""
    n = J_hat.shape[0]
    tree = set(tuple(sorted(e)) for e in tree_idx_edges)
    w_anat = {e: -J_hat[e[0], e[1]] for e in tree}
    L_anat = signed_laplacian([(i, j, w) for (i, j), w in w_anat.items()], n)
    M = J_hat - L_anat
    return L_anat, M, w_anat

def read_couplings(M, pairs):
    out = []
    for (i, j, lab) in pairs:
        v = M[i, j]
        phase = 'in-phase' if v < -1e-9 else ('anti-phase' if v > 1e-9 else 'none')
        out.append((lab, float(v), phase))
    return out

# --------------------------------------------------------------------------- #
#  SYNTHETIC ROUND-TRIP: plant a metaphor, recover it                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tree = [(IDX[a], IDX[b]) for a, b in KINEMATIC_TREE]
    limb_pairs = [
        (IDX["L_wrist"], IDX["R_wrist"], "arms  (L_wrist ~ R_wrist)"),
        (IDX["L_elbow"], IDX["R_elbow"], "arms  (L_elbow ~ R_elbow)"),
        (IDX["L_ankle"], IDX["R_ankle"], "legs  (L_ankle ~ R_ankle)"),
    ]
    def run(metaphor_sign, name):
        edges = tree_edges(0.15) + within_arm_edges(0.3) + cross_arm_edges(metaphor_sign, 1.2)
        J_true = precision_signed(edges, gamma=0.5)
        cov = covariance(J_true)                 # generate movement covariance
        J_hat = recover_precision(cov)           # recover precision from covariance
        L_anat, M, w = decompose(J_hat, tree)    # decompose
        print(f"\n[{name}]  planted metaphor: arms {'ANTI-phase' if metaphor_sign<0 else 'IN-phase'} (weight 1.2)")
        print(f"   recover exact?  max|J_hat - J_true| = {np.abs(J_hat-J_true).max():.2e}")
        # residual should vanish on tree edges, carry the metaphor off-tree
        tree_offdiag = max(abs(M[i,j]) for i,j in tree)
        print(f"   metaphor residual on TREE edges (should be ~0): {tree_offdiag:.2e}")
        for lab, v, ph in read_couplings(M, limb_pairs):
            print(f"   recovered {lab}:  M={v:+.3f}  -> {ph}")
    print("="*70); print("RECOVER-AND-DECOMPOSE round-trip"); print("="*70)
    run(-1, "walk-like (arms anti-phase)")
    run(+1, "jump-like (arms in-phase)")
    print("\n-> the off-tree residual M recovers the planted metaphor and flips sign\n   with the task, while the tree carries the anatomy. That is the model.")
