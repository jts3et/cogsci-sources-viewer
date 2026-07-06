#!/usr/bin/env bash
# Download and unzip the segmented Kinect skeletons of UMONS-TAICHI (153 MB).
# Tits, Laraba, Caulier, Tilmanne & Dutoit 2018, Data in Brief; CC BY-NC-SA 4.0.
# https://doi.org/10.1016/j.dib.2018.05.088   record: https://zenodo.org/records/2784581
set -e
cd "$(dirname "$0")"
url="https://zenodo.org/records/2784581/files/Segmented_Kinect.zip?download=1"
echo "downloading Segmented_Kinect.zip (153 MB) ..."
curl -sL "$url" -o Segmented_Kinect.zip
echo "unzipping ..."
unzip -q -o Segmented_Kinect.zip
echo "done -> Segmented_Kinect/  ($(ls Segmented_Kinect/*.txt | wc -l | tr -d ' ') gesture files)"
