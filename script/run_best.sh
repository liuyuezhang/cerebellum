#!/usr/bin/env bash

SEED=0

python -m main --granule rc --k 10  --n-hidden  500000 --lr 1e-6      --ltd ma --seed $SEED --wandb --save &
python -m main --granule rc --k 10  --n-hidden 1340000 --lr 3.731e-7  --ltd ma --seed $SEED --wandb --save &
