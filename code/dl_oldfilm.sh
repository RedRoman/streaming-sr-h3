#!/usr/bin/env bash
# Real degraded old-film frames, for a qualitative look at what our arm does to footage it was never
# trained for. github.com and codeload are reachable from this host (archive.org is DNS-poisoned and
# Google Drive / Dropbox / OneDrive all fail), so the RTN test set is the one that can actually be
# fetched without help.
#
# Written as a script, not passed inline: every inline background download today died with an empty log,
# and the one thing they shared was a curl -w format string full of % and {} going through the tmux
# session wrapper.
set -u
D=/cache/zihaohe/oldfilm
mkdir -p "$D"
cd "$D" || exit 1
for try in 1 2 3; do
  echo "attempt $try at $(date +%H:%M:%S)"
  curl -L --max-time 3600 -C - -o rtn.tar.gz \
    "https://gh-proxy.com/https://codeload.github.com/raywzy/Bringing-Old-Films-Back-to-Life/tar.gz/refs/heads/main" 2>&1 | tail -2
  sz=$(stat -c%s rtn.tar.gz 2>/dev/null || echo 0)
  echo "size now $sz"
  if [ "$sz" -gt 100000000 ]; then echo "COMPLETE $sz"; break; fi
  sleep 10
done
tar xzf rtn.tar.gz --wildcards '*/test_data/*' 2>/dev/null
mv Bringing-Old-Films-Back-to-Life-*/test_data real_rtn 2>/dev/null
for d in real_rtn/data_1/*/; do
  n=$(ls "$d" 2>/dev/null | wc -l)
  echo "  $(basename $d): $n frames"
done
python3 - <<'PY'
import glob
from PIL import Image
for d in sorted(glob.glob('/cache/zihaohe/oldfilm/real_rtn/data_1/*/')):
    fs = sorted(glob.glob(d + '*.png'))
    if not fs:
        continue
    im = Image.open(fs[0])
    print(d.split('/')[-2], im.size, im.mode, len(fs), 'frames')
PY
