#!/usr/bin/env bash
# fetch_cmu.sh -- download the CMU Graphics Lab MoCap trials used by fit_ggm.py.
# Data: mocap.cs.cmu.edu (free for research/education). Only the .amc motion files
# are needed; fit_ggm.py carries the skeleton topology internally.
set -euo pipefail
cd "$(dirname "$0")"
BASE="http://mocap.cs.cmu.edu/subjects"
# walk (subject 07), run (subject 09), jump (subject 13) -- two trials each
FILES="07/07_01.amc 07/07_02.amc 09/09_01.amc 09/09_02.amc 13/13_11.amc 13/13_13.amc"
for u in $FILES; do
  fn="$(basename "$u")"
  echo "fetching $fn"
  curl -fsS --max-time 60 "$BASE/$u" -o "$fn"
done
echo "done. now run:  python3 fit_ggm.py"
