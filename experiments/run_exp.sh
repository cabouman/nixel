#!/bin/bash
# Typical experiment commands.
# First:  conda activate nixel
# Then, from this experiments/ directory, copy whichever line(s) you need.

# single image (quick): pretrain, then reconstruct image 0
python pretrain.py --exp single
python reconstruct.py --exp single --image 0

# many images (long): pretrain in the background, watch it, reconstruct when done
nohup python pretrain.py --exp many > runs/many.log 2>&1 &
tail -f runs/many.log
python reconstruct.py --exp many --image 0
