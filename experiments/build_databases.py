"""
Build the training-image databases under img_data/.

Two databases, both reproducible from fixed seeds, so they can be rebuilt
identically on any machine (the image files themselves are gitignored):

  img_data/natural/   nat_<seed>.png        grayscale photo from Lorem Picsum
  img_data/phantom/   phantom_<seed>.png    rasterized random ellipse phantom
                      phantom_<seed>.json   the analytic ellipse parameters

The PNGs are for viewing; the phantom JSON preserves the exact continuous
(analytic) form used for training. Existing files are skipped unless --force.

Examples
--------
  python build_databases.py                      build both, 100 images each, 512x512
  python build_databases.py --which phantom      only the phantom database
  python build_databases.py --which natural      only the natural database
  python build_databases.py --num 200 --force    rebuild 200 each, overwriting
  python build_databases.py --size 1024          use a 1024x1024 array size
"""

import argparse
import glob
import hashlib
import io
import json
import os
import random
import urllib.request

import numpy as np
import torch
from PIL import Image

from linr import pixel_grid, eval_ellipses, random_phantom, shepp_logan
from _paths import NATURAL_DIR, PHANTOM_DIR


def _gray(seed, size):
    """Final (size,size) grayscale image for a seed, from cache or download.
    Returns (image, path, was_downloaded)."""
    path = os.path.join(NATURAL_DIR, f"nat_{seed}.png")
    if os.path.exists(path):
        return (Image.open(path).convert("L").resize((size, size), Image.LANCZOS),
                path, False)
    url = f"https://picsum.photos/seed/{seed}/{size}/{size}"
    req = urllib.request.Request(url, headers={"User-Agent": "linr-demo"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    return (Image.open(io.BytesIO(data)).convert("L").resize((size, size), Image.LANCZOS),
            path, True)


def build_natural(num, size, seed_start, force):
    """Collect `num` UNIQUE natural images. Picsum's seed->image map is not
    injective, so we skip any seed whose image content already appeared (hash of
    the grayscale array) and keep advancing the seed until `num` distinct images
    are collected. Deterministic given (num, size, seed_start)."""
    os.makedirs(NATURAL_DIR, exist_ok=True)
    if force:
        for f in glob.glob(os.path.join(NATURAL_DIR, "nat_*.png")):
            os.remove(f)

    seen = {}                      # content hash -> seed of the first occurrence
    accepted = 0
    seed = seed_start
    cap = seed_start + max(5 * num, num + 50)   # guard against a small Picsum pool
    while accepted < num and seed < cap:
        try:
            img, path, _ = _gray(seed, size)
        except Exception as e:
            print(f"  skip natural seed {seed}: {e}")
            seed += 1
            continue
        h = hashlib.md5(np.asarray(img, np.uint8).tobytes()).hexdigest()
        if h in seen:
            if os.path.exists(path):      # a known duplicate -> keep it out of the DB
                os.remove(path)
            seed += 1
            continue
        seen[h] = seed
        if not os.path.exists(path):
            img.save(path)
        accepted += 1
        if accepted % 25 == 0:
            print(f"  natural: {accepted}/{num} unique")
        seed += 1
    msg = f"natural: {accepted} unique images in {NATURAL_DIR}"
    if accepted < num:
        msg += f" (cap reached at seed {seed}; Picsum pool may be exhausted)"
    print(msg)


def build_phantom(num, size, seed_start, force):
    os.makedirs(PHANTOM_DIR, exist_ok=True)
    coords = pixel_grid(size, torch.device("cpu"))
    made = 0
    for s in range(seed_start, seed_start + num):
        png = os.path.join(PHANTOM_DIR, f"phantom_{s}.png")
        js = os.path.join(PHANTOM_DIR, f"phantom_{s}.json")
        if os.path.exists(png) and os.path.exists(js) and not force:
            continue
        ellipses = shepp_logan() if s == 0 else random_phantom(random.Random(s))
        img = eval_ellipses(coords, ellipses).clamp(0, 1).numpy()   # clamp for viewing
        Image.fromarray((img * 255).astype(np.uint8)).save(png)
        json.dump({"size": size, "ellipses": ellipses}, open(js, "w"), indent=1)
        made += 1
        if made % 25 == 0:
            print(f"  phantom: {made} generated")
    print(f"phantom: {made} written, {PHANTOM_DIR}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", choices=["natural", "phantom", "both"], default="both",
                   help="which database(s) to build (default: both)")
    p.add_argument("--num", type=int, default=200,
                   help="number of images per database (default: 100)")
    p.add_argument("--size", type=int, default=512,
                   help="array size N (images are N x N) (default: 512)")
    p.add_argument("--seed-start", type=int, default=0,
                   help="first seed; images use seeds seed_start .. seed_start+num-1")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing files instead of skipping them")
    args = p.parse_args()

    if args.which in ("natural", "both"):
        build_natural(args.num, args.size, args.seed_start, args.force)
    if args.which in ("phantom", "both"):
        build_phantom(args.num, args.size, args.seed_start, args.force)


if __name__ == "__main__":
    main()
