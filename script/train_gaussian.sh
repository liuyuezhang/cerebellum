#!/usr/bin/env bash

SEED=0

python -m main --env gaussian --granule fc --k 4   --n-hidden 5000   --lr 1e-3  --ltd none  --seed $SEED --wandb --save &
python -m main --env gaussian --granule fc --k 4   --n-hidden 5000   --lr 1e-3  --ltd ma    --seed $SEED --wandb --save &
python -m main --env gaussian --granule rc --k 4   --n-hidden 5000   --lr 1e-3  --ltd ma    --seed $SEED --wandb --save &
python -m main --env gaussian --granule rc --k 20  --n-hidden 5000   --lr 1e-3  --ltd ma    --seed $SEED --wandb --save &
python -m main --env gaussian --granule rc --k 100 --n-hidden 5000   --lr 1e-3  --ltd ma    --seed $SEED --wandb --save &
