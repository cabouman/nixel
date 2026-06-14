#!/bin/bash
# Typical experiment commands.
# First:  conda activate nixel
# Then, from this experiments/ directory:  bash run_exp.sh
# (or just copy whichever line you need)

# single image: pretrain, then reconstruct image 0
python pretrain.py --exp single
python reconstruct.py --exp single --image 0

# many images: pretrain, then reconstruct image 0
python pretrain.py --exp many
python reconstruct.py --exp many --image 0
