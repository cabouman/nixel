#!/bin/bash
set -e
cd "$(dirname "$0")"

python build_databases.py                       # build natural + phantom dbs (skips existing)


# Cold start a model
nohup python pretrain.py --exp best-v3 --name best-v3 > runs/pretrain.log 2>&1 &

# Improve a model
nohup python pretrain.py --exp default --name best_v2 --init-from best_model > runs/pretrain.log 2>&1 &

