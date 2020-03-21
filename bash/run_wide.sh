#!/usr/bin/env bash

python -m main --n-hidden 1000   --lr 5e-4    --wandb --save &
python -m main --n-hidden 2000   --lr 2.5e-4  --wandb --save &
python -m main --n-hidden 5000   --lr 1e-4    --wandb --save &
python -m main --n-hidden 10000  --lr 5e-5    --wandb --save &
python -m main --n-hidden 20000  --lr 2.5e-5  --wandb --save &
python -m main --n-hidden 50000  --lr 1e-5    --wandb --save &
python -m main --n-hidden 100000 --lr 5e-6    --wandb --save &
python -m main --n-hidden 200000 --lr 2.5e-6  --wandb --save &