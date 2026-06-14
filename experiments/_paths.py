"""Canonical data/output locations, anchored to the experiments/ directory so
they are the same no matter what working directory a script is run from."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "img_data")
NATURAL_DIR = os.path.join(IMG_DIR, "natural")
PHANTOM_DIR = os.path.join(IMG_DIR, "phantom")
RUNS_DIR = os.path.join(HERE, "runs")         # all generated experiment outputs
