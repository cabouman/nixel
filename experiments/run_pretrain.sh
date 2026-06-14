#!/bin/bash
# Pretrain models into runs/<name>/. Edit the configs.py templates or override with
# --set; --name sets the run-dir name so models aren't overwritten.
# First: conda activate nixel. Run from this experiments/ directory.

set -e
cd "$(dirname "$0")"

python build_databases.py                       # build natural + phantom dbs (skips existing)

python pretrain.py --exp single  --name single  # quick 1-image model -> runs/single/
python pretrain.py --exp default --name blend   # main blend model (long) -> runs/blend/

# background the long run instead of the line above, and resume if interrupted:
#   nohup python pretrain.py --exp default --name blend > runs/blend.log 2>&1 &
#   tail -f runs/blend.log
#   python pretrain.py --exp default --name blend --resume
