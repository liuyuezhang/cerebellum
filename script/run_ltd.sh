#!/usr/bin/env bash

SEED=0

python -m main --n-hidden 2000   --lr 2.5e-4  --ltd ma --seed $SEED --wandb --save &
python -m main --n-hidden 20000  --lr 2.5e-5  --ltd ma --seed $SEED --wandb --save &
python -m main --n-hidden 200000 --lr 2.5e-6  --ltd ma --seed $SEED --wandb --save &