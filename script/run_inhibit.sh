#!/usr/bin/env bash

SEED=0

python -m main --granule rc --k 200 --n-hidden 21818  --lr 2.292e-5  --ltd ma --golgi --seed $SEED --wandb --save &
python -m main --granule rc --k 10  --n-hidden 160000 --lr 3.125e-6  --ltd ma --golgi --seed $SEED --wandb --save &
python -m main --granule rc --k 4   --n-hidden 200000 --lr 2.5e-6    --ltd ma --golgi --seed $SEED --wandb --save &