#!/usr/bin/env bash

SEED=0
ATTACK=pgd
ENV=gaussian2

python -m attack --attack $ATTACK --env $ENV --granule fc --k 4   --n-hidden 2000     --lr 2.5e-4   --ltd none  --seed $SEED --wandb --save &
python -m attack --attack $ATTACK --env $ENV --granule fc --k 4   --n-hidden 126000   --lr 3.97e-6  --ltd none  --seed $SEED --wandb --save &
python -m attack --attack $ATTACK --env $ENV --granule fc --k 4   --n-hidden 42000    --lr 1.19e-5  --ltd none  --seed $SEED --wandb --save &
python -m attack --attack $ATTACK --env $ENV --granule fc --k 4   --n-hidden 9692     --lr 5.16e-5  --ltd none  --seed $SEED --wandb --save &

python -m attack --attack $ATTACK --env $ENV --granule fc --k 4   --n-hidden 2000     --lr 2.5e-4   --ltd ma    --seed $SEED --wandb --save &
python -m attack --attack $ATTACK --env $ENV --granule rc --k 4   --n-hidden 2000     --lr 2.5e-4   --ltd ma    --seed $SEED --wandb --save &
python -m attack --attack $ATTACK --env $ENV --granule rc --k 20  --n-hidden 2000     --lr 2.5e-4   --ltd ma    --seed $SEED --wandb --save &
python -m attack --attack $ATTACK --env $ENV --granule rc --k 100 --n-hidden 2000     --lr 2.5e-4   --ltd ma    --seed $SEED --wandb --save &

python -m attack --attack $ATTACK --env $ENV --granule rc --k 4   --n-hidden 126000   --lr 3.97e-6  --ltd ma    --seed $SEED --wandb --save &
python -m attack --attack $ATTACK --env $ENV --granule rc --k 20  --n-hidden 42000    --lr 1.19e-5  --ltd ma    --seed $SEED --wandb --save &
python -m attack --attack $ATTACK --env $ENV --granule rc --k 100 --n-hidden 9692     --lr 5.16e-5  --ltd ma    --seed $SEED --wandb --save &
