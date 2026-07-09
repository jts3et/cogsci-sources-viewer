# Model data — inputs and outputs

This folder collects the coordination model's **inputs** (the small ones) and its
**outputs** (the figures). The scripts live one level up in `model/`.

## Outputs — `figures/`

Every figure the model produces, committed here:

| file | produced by | data |
|------|-------------|------|
| `fig_joint_graph.png` | `generate_joint_graph.py` | none (model geometry) |
| `fig_two_body.png`, `fig_two_body_skeletons.png` | `two_body.py` | none (synthetic) |
| `fig_frustration.png` | `frustration.py` | none (synthetic) |
| `fig_cmu_ggm.png` | `fit_ggm.py` | CMU mocap (below) |
| `fig_relative_phase.png` | `relative_phase.py` | CMU mocap (below) |
| `fig_qualisys.png` | `fit_qualisys.py` | UMONS-TAICHI Qualisys (below) |
| `fig_umons.png` | `fit_umons.py` | UMONS-TAICHI Kinect (below) |

Every empirical figure has been verified to regenerate byte-identically from the
raw data, and every published statistic reproduces exactly.

## Inputs

### CMU Graphics Lab Motion Capture — `cmu/` (committed)

Six trials, ~1.4 MB total, small enough to live in the repo: walk (`07_01`, `07_02`),
run (`09_01`, `09_02`), jump (`13_11`, `13_13`). Free for research and education,
from <http://mocap.cs.cmu.edu>. Re-fetch with `bash ../fetch_cmu.sh`.

### UMONS-TAICHI — NOT committed (too large for GitHub)

The tai-chi recordings exceed GitHub's 100 MB-per-file limit, so they are **not**
stored here. They are public on Zenodo and fetched on demand:

- **Kinect skeletons** — `Segmented_Kinect.zip` (160 MB) → `bash ../fetch_umons.sh`
- **Qualisys markers** — `Segmented_TSV.zip` (3.4 GB) → `bash ../fetch_qualisys.sh`

Tits, Laraba, Caulier, Tilmanne & Dutoit (2018), *Data in Brief*,
<https://doi.org/10.1016/j.dib.2018.05.088>; record
<https://zenodo.org/records/2784581>, CC BY-NC-SA 4.0.

## Reproduce everything

```
cd ..                       # into model/
bash fetch_cmu.sh           # CMU mocap (also committed in cmu/)
python3 fit_ggm.py          # -> fig_cmu_ggm.png
python3 relative_phase.py   # -> fig_relative_phase.png
python3 two_body.py         # -> fig_two_body.png, fig_two_body_skeletons.png
python3 frustration.py      # -> fig_frustration.png
python3 generate_joint_graph.py   # -> fig_joint_graph.png
bash fetch_umons.sh && python3 fit_umons.py       # -> fig_umons.png (160 MB download)
bash fetch_qualisys.sh && python3 fit_qualisys.py # -> fig_qualisys.png (3.4 GB download)
```

The scripts currently write figures into `model/`; the copies here are the committed
record. The site serves its own copies from the repo root.
