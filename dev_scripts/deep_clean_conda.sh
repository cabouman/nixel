#!/bin/bash
# Deactivate all conda environments and wipe conda/pip caches and user-site
# packages for a truly clean reinstall. Run with: source deep_clean_conda.sh

while [ ${#CONDA_DEFAULT_ENV} -gt 0 ]; do
    conda deactivate
done

rm -rf ~/.conda/* ~/.cache/conda/* ~/.cache/pip/* ~/.local/lib/python*
