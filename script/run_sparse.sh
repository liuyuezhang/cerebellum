#!/usr/bin/env bash

SEED=0

python -m main --granule fc --k 784 --n-hidden 5970   --lr 8.375e-5  --ltd ma --seed $SEED --wandb --save &
python -m main --granule rc --k 200 --n-hidden 21818  --lr 2.292e-5  --ltd ma --seed $SEED --wandb --save &
python -m main --granule rc --k 100 --n-hidden 40000  --lr 1.25e-5   --ltd ma --seed $SEED --wandb --save &
python -m main --granule rc --k 50  --n-hidden 68571  --lr 7.292e-6  --ltd ma --seed $SEED --wandb --save &
python -m main --granule rc --k 20  --n-hidden 120000 --lr 4.167e-6  --ltd ma --seed $SEED --wandb --save &
python -m main --granule rc --k 10  --n-hidden 160000 --lr 3.125e-6  --ltd ma --seed $SEED --wandb --save &
python -m main --granule rc --k 4   --n-hidden 200000 --lr 2.5e-6    --ltd ma --seed $SEED --wandb --save &
python -m main --granule rc --k 2   --n-hidden 218182 --lr 2.292e-6  --ltd ma --seed $SEED --wandb --save &
python -m main --granule rc --k 1   --n-hidden 228571 --lr 2.186e-6  --ltd ma --seed $SEED --wandb --save &