"""
signed_laplacian.py -- the unification Cody pointed at, now the model's single source.

Two precision builders once disagreed in FORM while agreeing in content:
  * a clique Laplacian for the metaphor, built in-phase only, and
  * two_body's relational 1/2 w (a x_i - b x_j)^2 terms (the signed "Ising" bit).

They are ONE object: a SIGNED graph Laplacian with the absolute-degree (|W|)
convention, which now lives in taichi_model (signed_laplacian, precision_signed,
and precision_matrix's metaphor_sign) as the single builder. A metaphor is a set of
SIGNED edges linking joint-communities:
    edge weight  > 0  ->  in-phase  (ordinary Laplacian block  D - W)
    edge weight  < 0  ->  anti-phase (signless block           D + |W|)
    J = gamma*I + signed_laplacian(anatomy[+] + metaphor[signed]).
This module rebuilds two_body's relational terms as signed edges and proves the two
forms give an identical J. No spin model imported; in/anti-phase is edge sign,
frustration is signed-graph imbalance.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The builder now lives in taichi_model as the single source; re-exported here.
from taichi_model import N, IDX, baseline_edges, signed_laplacian, precision_signed

# ---- rebuild the two_body relational terms as SIGNED EDGES ------------------ #
def tree_edges(w_base):
    be, bw = baseline_edges(w_base)
    return [(i, j, w) for (i, j), w in zip(be, bw)]           # anatomy: positive
def cross_arm_edges(sign, w):
    pairs = [("L_shoulder","R_shoulder"),("L_elbow","R_elbow"),("L_wrist","R_wrist")]
    return [(IDX[a], IDX[b], sign*w) for a, b in pairs]        # metaphor: signed
def within_arm_edges(w):
    out=[]
    for s in ("L","R"):
        j=[IDX[f"{s}_shoulder"],IDX[f"{s}_elbow"],IDX[f"{s}_wrist"]]
        out += [(j[0],j[1],w),(j[1],j[2],w)]                  # synergy: positive
    return out

if __name__ == "__main__":
    # PROVE it reproduces two_body.relational_precision EXACTLY for the teacher body
    from two_body import relational_precision, cross_arm_terms, within_arm_terms, arm_order_parameter
    from taichi_model import covariance
    be, bw = baseline_edges(0.15)
    idio_terms = cross_arm_terms(-1, 1.2) + within_arm_terms(0.3)     # teacher
    J_old = relational_precision(be, bw, idio_terms, gamma=0.50)

    edges = tree_edges(0.15) + cross_arm_edges(-1, 1.2) + within_arm_edges(0.3)
    J_new = precision_signed(edges, gamma=0.50)

    print("max |J_signed - J_relational| =", np.abs(J_new - J_old).max())
    print("identical J ?", np.allclose(J_new, J_old))
    rT = arm_order_parameter(covariance(J_old))
    rN = arm_order_parameter(covariance(J_new))
    print(f"teacher order parameter  relational={rT:+.4f}   signed-laplacian={rN:+.4f}")
