#!/usr/bin/env bash

ATTACK=fgsm
SEED=0

python -m attack --attack $ATTACK --n-hidden 1000   --lr 5e-4    --seed $SEED --wandb &
python -m attack --attack $ATTACK --n-hidden 2000   --lr 2.5e-4  --seed $SEED --wandb &
python -m attack --attack $ATTACK --n-hidden 5000   --lr 1e-4    --seed $SEED --wandb &
python -m attack --attack $ATTACK --n-hidden 10000  --lr 5e-5    --seed $SEED --wandb &
python -m attack --attack $ATTACK --n-hidden 20000  --lr 2.5e-5  --seed $SEED --wandb &
python -m attack --attack $ATTACK --n-hidden 50000  --lr 1e-5    --seed $SEED --wandb &
python -m attack --attack $ATTACK --n-hidden 100000 --lr 5e-6    --seed $SEED --wandb &
python -m attack --attack $ATTACK --n-hidden 200000 --lr 2.5e-6  --seed $SEED --wandb &