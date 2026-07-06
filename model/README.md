# The coordination model — reproducible fit

Proof-of-concept fit of a Gaussian graphical model (GGM) of body coordination to
real CMU motion-capture, backing the claims on the site's
[Modelling](../modelling.html) tab.

## What it tests

Three premises the model rests on, checked against real human movement:

- **(A) Dimensionality** — is real joint-angle covariance low-rank (a few synergies)?
- **(B) Structure** — do the GGM partial correlations concentrate on *anatomically
  adjacent* joints, i.e. does the precision matrix recover the kinematic tree the
  model's graph-Laplacian is built from?
- **(C) Task-dependence** — does the *sign* of limb coupling flip across tasks
  (walk/run anti-phase → jump in-phase) at nearly fixed dimensionality? This is the
  real-data proxy for "same collapse, different relational content."

## Run it

```bash
bash fetch_cmu.sh      # downloads 6 .amc trials from mocap.cs.cmu.edu (~1 MB)
python3 fit_ggm.py     # prints the results, writes fig_cmu_ggm.png
```

Requires `numpy` and `matplotlib`. The figure published on the Modelling tab lives at
the repo root (`../fig_cmu_ggm.png`); re-running writes a fresh copy alongside the
script.

## What it reports (as of the committed run)

| task | frames | PR(raw cov)/50 | PR(corr)/50 | anat. AUC |
|------|-------:|---------------:|------------:|----------:|
| walk | 645 | 2.74 | 4.78 | 0.798 |
| run  | 278 | 2.98 | 5.05 | 0.712 |
| jump | 854 | 2.65 | 4.96 | 0.817 |

Signed limb coordination (matched flexion axis): legs L/R = −0.92 (walk), −0.89 (run),
**+1.00 (jump)**; arms L/R = −0.95, −0.85, **+0.93**. Same ~5 effective DOF, opposite sign.

## Notes / limits

- `fit_ggm.py` is self-contained (it re-implements PR, parsing, and the topology); it
  does **not** import the group's core model files.
- Sign is axis-convention dependent — only the *relative* sign structure is interpretable.
- One subject, two trials per task — a proof of concept, not an estimate with error bars.
- Data provenance: CMU Graphics Lab Motion Capture Database, mocap.cs.cmu.edu.

## Step 1 — two-body convergence

`two_body.py` runs the first step of the intended outcome: it shows that a
metaphor drives two differently built bodies onto the same relative-phase
coordination, where a joint-angle instruction cannot. No new data; it runs on
the same model.

```bash
python3 two_body.py     # needs numpy, matplotlib, and taichi_model.py alongside
```

Result (as committed):

- Teacher trained to an anti-phase (mirror) arm coordination, r_T = −0.69.
- Two learners with different anatomy start in-phase (r = +0.61, +0.56), a gap of ~1.3.
- Under the shared metaphor the gap closes to ~0.1 (learners reach r = −0.53, −0.71).
- The joint-angle channel sets a mean pose and never touches J, so its gap stays ~1.27 at every strength.

Figures: `../fig_two_body.png` (convergence curve) and `../fig_two_body_skeletons.png`
(one joint-angle target reaches different places on different bodies; the metaphor
imposes the same mirror coordination at every scale).

## Section 2 — fit to real Taijiquan motion (UMONS-TAICHI)

`fit_umons.py` runs the same fit on real Taijiquan, the movement this project is
actually about. UMONS-TAICHI (Tits et al. 2018, Data in Brief,
[10.1016/j.dib.2018.05.088](https://doi.org/10.1016/j.dib.2018.05.088);
[Zenodo](https://zenodo.org/records/2784581), CC BY-NC-SA 4.0) is Kinect V2
motion capture of 12 performers doing 13 techniques, with a 0–10 skill score per
performer from three judges. Each frame becomes 10 joint angles; the fit runs on
the 1,793 clips with ≥60 tracked frames.

```bash
bash fetch_umons.sh && python3 fit_umons.py   # downloads 153 MB, writes ../fig_umons.png
```

Result (as committed):

- **(A) Dimensionality** — all 13 techniques collapse 10 angles to ~4 effective
  axes (Kinect median PR 3.85). The collapse holds on real tai-chi.
- **(B) Coupling graph (Kinect)** — noisy; recovers the anatomical tree only at
  chance (AUC ~0.53). Read the clean version on the Qualisys markers instead
  (`fit_qualisys.py`), where AUC is 0.72–0.78.
- **(C) A Kinect-only finding that did NOT replicate** — on the 30 Hz Kinect,
  per-performer lower-body coupling strength appeared to track skill at Spearman
  ρ = +0.71. The high-resolution Qualisys markers **reverse** this (ρ = −0.51),
  and the two sensors anti-correlate per-performer (r = −0.33). So it is a
  sensor artifact, not a property of the movement. **See `fit_qualisys.py` for
  the corrected result.** Effective dimensionality does not track skill on
  either sensor.

Self-contained except for `numpy`/`scipy`/`matplotlib`; does not import the
group's core model files. Writes `umons_perP.npy` (per-performer summary the
Qualisys figure overlays) and `fig_umons.png`.

## Section 2 (continued) — the high-resolution check (Qualisys markers)

`fit_qualisys.py` runs the same ten-angle fit on the optical marker capture from
the *same sessions*: Qualisys, 68 markers on real anatomical landmarks, 179 Hz,
hand-cleaned. Reads `Segmented_TSV.zip` directly (`bash fetch_qualisys.sh`,
3.6 GB). Needs `pandas` in addition to `numpy`/`scipy`/`matplotlib`.

```bash
bash fetch_qualisys.sh && python3 fit_qualisys.py   # writes ../fig_qualisys.png
```

Why two instruments: the Kinect infers joints markerlessly; the Qualisys puts
markers on bone. Running the fit on both separates movement from sensor.

Result (as committed):

- **(A) Collapse holds on both** — Qualisys per-clip PR ~2.7 of 10; Kinect ~3.85.
- **(B) Clean graph** — partial-correlation graph recovers the anatomical tree at
  AUC 0.72 (expert) / 0.78 (novice); strongest edges are the leg chains, plus a
  cross-body arm diagonal (elbow ↔ opposite shoulder).
- **(C) The real skill signal is timing, not amount** — bilateral arm
  phase-locking (how steadily the two arms hold a fixed relative phase, 179 Hz)
  tracks judged skill at Spearman ρ = +0.68 (p = 0.015, leave-one-out
  [+0.61, +0.77]). The static leg-coupling of the Kinect reverses here. This is
  the Haken–Kelso–Bunz order-parameter prediction: skill = a steadier relative
  phase, visible only at high frame rate.

Figure `../fig_qualisys.png` (4 panels: arm phase-locking vs skill; the
two-sensor leg-coupling disagreement; dimensionality on both sensors; the
Qualisys expert coupling matrix). Run `fit_umons.py` first so the cross-sensor
panels can overlay the Kinect numbers from `umons_perP.npy`.

## Section 1(D) — measured relative phase

`relative_phase.py` recovers the actual relative phase between paired limbs from
the CMU time series with a Hilbert transform (needs `scipy`), not just the sign
of coordination. Walk legs 181° and run legs 170° (anti-phase), jump legs 3°
(in-phase), phase-locking 0.9–0.99. Figure `../fig_relative_phase.png`.

```bash
bash fetch_cmu.sh && python3 relative_phase.py
```

## Step 3 — frustration

`frustration.py` sweeps the conflict between an in-phase and an anti-phase image
on one arm pair. At conflict ½ the free coordinated mode's variance drops from
1.88 to 0.28 and the arm correlation passes through 0.00: two images cancel
rather than average. No new data. Figure `../fig_frustration.png`.

```bash
python3 frustration.py
```
