#!/usr/bin/env bash

SEED=0

#python -m main --epoch 10 --env cifar10 --granule fc       --ltd none --n-hidden 1166   --lr 4.288e-4   --seed $SEED --wandb --save &
#python -m main --epoch 10 --env cifar10 --granule fc       --ltd ma   --n-hidden 1166   --lr 4.288e-4   --seed $SEED --wandb --save &
#python -m main --epoch 10 --env cifar10 --granule fc       --ltd none --n-hidden 5000   --lr 1e-4       --seed $SEED --wandb --save &
#python -m main --epoch 10 --env cifar10 --granule fc       --ltd ma   --n-hidden 5000   --lr 1e-4       --seed $SEED --wandb --save &
python -m main --epoch 10 --env cifar10 --granule rc --k 4 --ltd none --n-hidden 200000 --lr 2.5e-6     --seed $SEED --wandb --save &
python -m main --epoch 10 --env cifar10 --granule rc --k 4 --ltd ma   --n-hidden 200000 --lr 2.5e-6     --seed $SEED --wandb --save &