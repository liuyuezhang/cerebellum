#!/usr/bin/env bash

SEED=0

GRANULE=rc
python -m main --env cifar10 --granule $GRANULE --k 3072 --n-hidden 1563   --lr 3.199e-4  --ltd ma  --seed $SEED --wandb --save &
python -m main --env cifar10 --granule $GRANULE --k 1200 --n-hidden 3934   --lr 1.271e-4  --ltd ma  --seed $SEED --wandb --save &
python -m main --env cifar10 --granule $GRANULE --k 400  --n-hidden 11429  --lr 4.375e-5  --ltd ma  --seed $SEED --wandb --save &
python -m main --env cifar10 --granule $GRANULE --k 100  --n-hidden 40000  --lr 1.25e-5   --ltd ma  --seed $SEED --wandb --save &
python -m main --env cifar10 --granule $GRANULE --k 50   --n-hidden 68571  --lr 7.292e-6  --ltd ma  --seed $SEED --wandb --save &
python -m main --env cifar10 --granule $GRANULE --k 20   --n-hidden 120000 --lr 4.167e-6  --ltd ma  --seed $SEED --wandb --save &
python -m main --env cifar10 --granule $GRANULE --k 10   --n-hidden 160000 --lr 3.125e-6  --ltd ma  --seed $SEED --wandb --save &
python -m main --env cifar10 --granule $GRANULE --k 4    --n-hidden 200000 --lr 2.5e-6    --ltd ma  --seed $SEED --wandb --save &
python -m main --env cifar10 --granule $GRANULE --k 2    --n-hidden 218182 --lr 2.292e-6  --ltd ma  --seed $SEED --wandb --save &
python -m main --env cifar10 --granule $GRANULE --k 1    --n-hidden 228571 --lr 2.186e-6  --ltd ma  --seed $SEED --wandb --save &

python -m main --env cifar10 --granule rc       --k 4    --n-hidden 200000 --lr 2.5e-6    --ltd none --seed $SEED --wandb --save &
python -m main --env cifar10 --granule fc                --n-hidden 1563   --lr 3.199e-4  --ltd ma   --seed $SEED --wandb --save &
python -m main --env cifar10 --granule fc                --n-hidden 1563   --lr 3.199e-4  --ltd none --seed $SEED --wandb --save &