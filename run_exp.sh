#!/bin/bash
# run_exp.sh -- typical nixel experiment sequences (a runnable cheat-sheet).
#
# Activate the env first:   conda activate nixel
# (or prefix with PY=...:    PY=/opt/anaconda3/envs/nixel/bin/python ./run_exp.sh single)
#
# Usage:  ./run_exp.sh <command> [args]
#   single            pretrain (1 image) + reconstruct image 0          (quick, local)
#   many              pretrain the many-image decoder in the foreground  (LONG)
#   many-bg           same, but in the background, logging to runs/many.log
#   recon [imgs...]   reconstruct images with the many decoder (default: 0 5 42)
#   all               single, then many (foreground), then recon         (LONG)

set -euo pipefail

# cd to the repo root (this script's directory), robust to where it's called from
cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

PY="${PY:-python}"
if ! "$PY" -c "import linr" >/dev/null 2>&1; then
  echo "ERROR: '$PY' cannot import linr. Activate the env:  conda activate nixel"
  echo "       or run:  PY=/opt/anaconda3/envs/nixel/bin/python ./run_exp.sh $*"
  exit 1
fi

single() {
  echo "==> [single] pretrain (1 image, progressive) -> runs/single/"
  "$PY" experiments/pretrain.py --exp single
  echo "==> [single] reconstruct image 0 (warm-start adapt-theta) -> runs/single/recon/"
  "$PY" experiments/reconstruct.py --exp single --image 0
}

many() {
  echo "==> [many] pretrain (long; foreground) -> runs/many/"
  "$PY" experiments/pretrain.py --exp many
}

many_bg() {
  mkdir -p runs
  echo "==> [many] pretrain in the background -> runs/many.log"
  nohup "$PY" experiments/pretrain.py --exp many > runs/many.log 2>&1 &
  echo "    started PID $!   (watch:  tail -f runs/many.log)"
}

recon() {
  local imgs=("$@"); [ ${#imgs[@]} -eq 0 ] && imgs=(0 5 42)
  for i in "${imgs[@]}"; do
    echo "==> [many] reconstruct image $i -> runs/many/recon/img${i}_adapt.png"
    "$PY" experiments/reconstruct.py --exp many --image "$i"
  done
}

cmd="${1:-}"; shift || true
case "$cmd" in
  single)   single ;;
  many)     many ;;
  many-bg)  many_bg ;;
  recon)    recon "$@" ;;
  all)      single; many; recon ;;
  *) echo "usage: ./run_exp.sh {single|many|many-bg|recon [imgs...]|all}"; exit 1 ;;
esac
