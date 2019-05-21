#!/usr/bin/env bash

#for final_lr in 1.0 0.2 0.1 0.05 0.01; do
#  for lr in 0.02 0.01 0.001 0.0001; do
#    python train.py --optimizer adabound --model bnsrcnn --lr ${lr} --final_lr ${final_lr} --batch 128 --loss mse-clop
#  done
#done#
for opt in 'sgd' 'adam'; do
for lr in 0.001 0.0005 0.0001; do
  python train.py --optimizer ${opt} --model espcn --lr ${lr} --batch 128 --dataset bsds
done
done

for final_lr in 1.0 0.2 0.1 0.05 0.01; do
  for lr in 0.02 0.01 0.001 0.0001; do
    python train.py --optimizer adabound --model espcn --lr ${lr} --final_lr ${final_lr} --batch 128 --dataset bsds
  done
done
#
#for opt in 'sgd' 'adam'; do
#  for model in 'bnsrcnn' 'srcnn'; do
#    for lr in 0.01 0.001 0.0001; do
#      python train.py --optimizer ${opt} --model ${model} --lr ${lr} --batch 128 --loss mse-clop
#    done
#  done
#done
