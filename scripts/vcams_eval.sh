#!/bin/bash

MODEL_PATH=("nerf_synthetic/ship_v10" "m360/kitchen_v10" "tant/train_v20")
SEEDS=(0 29506 1000 518 7463)

for seed in "${SEEDS[@]}"; do
  for path in  "${MODEL_PATH[@]}"; do
    echo "Running script with seed: $seed and path: $path"
    python render_uncertainty_w_Vcams_v2.py -m ./output/$path \
          --render_vcam --n_vcam 6 --seed $seed --r_scale 0.05 0.1 0.25 0.5
    done
  done