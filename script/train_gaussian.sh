#!/usr/bin/env bash

SEED=0
EPOCH=200
ENV=gaussian2

python -m main --env $ENV --epoch $EPOCH --granule fc --k 4   --n-hidden 5000   --lr 1e-4  --ltd none  --seed $SEED --wandb --save &
python -m main --env $ENV --epoch $EPOCH --granule fc --k 4   --n-hidden 5000   --lr 1e-4  --ltd ma    --seed $SEED --wandb --save &
python -m main --env $ENV --epoch $EPOCH --granule rc --k 4   --n-hidden 5000   --lr 1e-4  --ltd ma    --seed $SEED --wandb --save &
python -m main --env $ENV --epoch $EPOCH --granule rc --k 20  --n-hidden 5000   --lr 1e-4  --ltd ma    --seed $SEED --wandb --save &
python -m main --env $ENV --epoch $EPOCH --granule rc --k 100 --n-hidden 5000   --lr 1e-4  --ltd ma    --seed $SEED --wandb --save &
