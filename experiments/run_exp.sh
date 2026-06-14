#!/bin/bash
# Reproduce the full pipeline end to end (single + many cases).
#
#   conda activate nixel
#   bash run_exp.sh            # runs as-is (it cds to its own dir)
#
# or copy/paste any line individually. The 'many' pretrain runs in the FOREGROUND
# so the reconstruct that follows waits for it to finish. (To background it
# instead, see the commented nohup variant at the bottom.)

set -e
cd "$(dirname "$0")"          # run from the experiments/ directory

# 1. build the image databases (natural downloads from Picsum; existing files skipped)
python build_databases.py

# 2. single image (quick): pretrain, then reconstruct image 0
python pretrain.py --exp single
python reconstruct.py --exp single --image 0

# 3. many images (long): pretrain, then reconstruct a natural and a phantom image
python pretrain.py --exp many
python reconstruct.py --exp many --image 0                      # natural image 0 (held out)
python reconstruct.py --exp many --database phantom --image 0   # phantom image 0

# --- alternative: background the long 'many' pretrain, watch it, reconstruct later ---
#   nohup python pretrain.py --exp many > runs/many.log 2>&1 &
#   tail -f runs/many.log
#   python reconstruct.py --exp many --image 0
