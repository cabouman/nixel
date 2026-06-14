#!/bin/bash
# Reconstruct images with a saved model (a runs/<name>/ dir from pretrain).
# Point --model at the model dir; --image picks the image.
# First: conda activate nixel. Run from this experiments/ directory.

set -e
cd "$(dirname "$0")"

python recon.py --model blend --image 0 --set recon_steps=1000                    # natural image 0 (held out)
python recon.py --model blend --database phantom --image 0 --set recon_steps=1000 # phantom 0 = Shepp-Logan (held out)

# fewer iterations / a different LR, without editing configs.py (--set any field):
#   python recon.py --model blend --database phantom --image 0 --set recon_steps=2000
