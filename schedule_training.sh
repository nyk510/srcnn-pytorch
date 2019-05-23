#!/usr/bin/env bash

for scale in 2 3 4 5; do
for lr in 0.01 0.001 0.0001; do
for final_lr in 1.0 0.2 0.1 0.05 0.01; do
    python train.py \
        --optimizer adabound \
        --model espcn \
        --lr ${lr} \
        --final_lr ${final_lr} \
        --batch 128 \
        --dataset bsds \
        --upscale ${scale}
  done
done
done