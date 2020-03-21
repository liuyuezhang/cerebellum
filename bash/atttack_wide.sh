#!/usr/bin/env bash

python -m attack --attack fgsm --n-hidden 1000   --lr 5e-4    --wandb &
python -m attack --attack fgsm --n-hidden 2000   --lr 2.5e-4  --wandb &
python -m attack --attack fgsm --n-hidden 5000   --lr 1e-4    --wandb &
python -m attack --attack fgsm --n-hidden 10000  --lr 5e-5    --wandb &
python -m attack --attack fgsm --n-hidden 20000  --lr 2.5e-5  --wandb &
python -m attack --attack fgsm --n-hidden 50000  --lr 1e-5    --wandb &
python -m attack --attack fgsm --n-hidden 100000 --lr 5e-6    --wandb &
python -m attack --attack fgsm --n-hidden 200000 --lr 2.5e-6  --wandb &