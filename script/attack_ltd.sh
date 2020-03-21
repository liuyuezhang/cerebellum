#!/usr/bin/env bash

ATTACK=fgsm
SEED=0

python -m attack --attack $ATTACK --n-hidden 2000   --lr 2.5e-4  --ltd ma --seed $SEED --wandb &
python -m attack --attack $ATTACK --n-hidden 20000  --lr 2.5e-5  --ltd ma --seed $SEED --wandb &
python -m attack --attack $ATTACK --n-hidden 200000 --lr 2.5e-6  --ltd ma --seed $SEED --wandb &