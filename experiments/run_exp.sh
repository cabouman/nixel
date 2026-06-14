#!/bin/bash
# Typical experiment commands.
# First:  conda activate nixel
# Then, from the repo root:  bash experiments/run_exp.sh
# (or just copy whichever line you need)

# single image: pretrain, then reconstruct image 0
python experiments/pretrain.py --exp single
python experiments/reconstruct.py --exp single --image 0

# many images: pretrain, then reconstruct image 0
python experiments/pretrain.py --exp many
python experiments/reconstruct.py --exp many --image 0
