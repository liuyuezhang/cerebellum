#!/usr/bin/env bash

SEED=0
ATTACK=fgsm
ENV=gaussian2

python -m attack --attack $ATTACK --env $ENV --granule fc --k 4   --n-hidden 5000   --lr 1e-4  --ltd none  --seed $SEED --wandb --save &
python -m attack --attack $ATTACK --env $ENV --granule fc --k 4   --n-hidden 5000   --lr 1e-4  --ltd ma    --seed $SEED --wandb --save &
python -m attack --attack $ATTACK --env $ENV --granule rc --k 4   --n-hidden 5000   --lr 1e-4  --ltd ma    --seed $SEED --wandb --save &
python -m attack --attack $ATTACK --env $ENV --granule rc --k 20  --n-hidden 5000   --lr 1e-4  --ltd ma    --seed $SEED --wandb --save &
python -m attack --attack $ATTACK --env $ENV --granule rc --k 100 --n-hidden 5000   --lr 1e-4  --ltd ma    --seed $SEED --wandb --save &
