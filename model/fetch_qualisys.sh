#!/usr/bin/env bash
# Download the segmented Qualisys optical marker capture of UMONS-TAICHI (3.6 GB).
# Tits, Laraba, Caulier, Tilmanne & Dutoit 2018, Data in Brief; CC BY-NC-SA 4.0.
# https://doi.org/10.1016/j.dib.2018.05.088   record: https://zenodo.org/records/2784581
# fit_qualisys.py reads the .zip directly, so no unzip step is needed.
set -e
cd "$(dirname "$0")"
url="https://zenodo.org/records/2784581/files/Segmented_TSV.zip?download=1"
echo "downloading Segmented_TSV.zip (3.6 GB) ..."
curl -L "$url" -o Segmented_TSV.zip
echo "done -> Segmented_TSV.zip  (fit_qualisys.py reads it directly)"
