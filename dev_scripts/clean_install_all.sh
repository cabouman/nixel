#!/bin/bash
# Create the `nixel` conda environment and editable-install the package + dev
# tools. Run from anywhere:  source dev_scripts/clean_install_all.sh
# For a really clean slate first run:  source dev_scripts/deep_clean_conda.sh
NAME="nixel"
PYTHON_VERSION="3.11"

# Move to the repo root (parent of dev_scripts), robust to run-or-source / bash-or-zsh.
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR/.." || { echo "Could not cd to repo root"; return 1 2>/dev/null || exit 1; }

if [ ! -f pyproject.toml ]; then
  echo "ERROR: pyproject.toml not found in $(pwd) -- run from the nixel repo."
  return 1 2>/dev/null || exit 1
fi

# Deactivate any active conda environments
while [ ${#CONDA_DEFAULT_ENV} -gt 0 ]; do conda deactivate; done

# Recreate the environment
yes | conda remove --name "$NAME" --all 2>/dev/null
yes | conda create -n "$NAME" python="$PYTHON_VERSION"
conda activate "$NAME"

# Editable install of the `linr` package + pytest (pulls torch, numpy, matplotlib, pillow).
# On a CUDA host, install the matching torch build first, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"

echo
echo "Done. Use:  conda activate $NAME"
echo "Then, from the experiments/ directory, e.g.  python pretrain.py --exp single"
echo
