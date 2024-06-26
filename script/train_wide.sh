#!/usr/bin/env bash

ENV=mnist
EPOCH=10
LTD=none
SEED=0

python -m main --env $ENV --epoch $EPOCH --n-hidden 1000   --lr 5e-4    --ltd $LTD --seed $SEED --wandb --save &
python -m main --env $ENV --epoch $EPOCH --n-hidden 2000   --lr 2.5e-4  --ltd $LTD --seed $SEED --wandb --save &
python -m main --env $ENV --epoch $EPOCH --n-hidden 5000   --lr 1e-4    --ltd $LTD --seed $SEED --wandb --save &
python -m main --env $ENV --epoch $EPOCH --n-hidden 10000  --lr 5e-5    --ltd $LTD --seed $SEED --wandb --save &
python -m main --env $ENV --epoch $EPOCH --n-hidden 20000  --lr 2.5e-5  --ltd $LTD --seed $SEED --wandb --save &
python -m main --env $ENV --epoch $EPOCH --n-hidden 50000  --lr 1e-5    --ltd $LTD --seed $SEED --wandb --save &
python -m main --env $ENV --epoch $EPOCH --n-hidden 100000 --lr 5e-6    --ltd $LTD --seed $SEED --wandb --save &
python -m main --env $ENV --epoch $EPOCH --n-hidden 200000 --lr 2.5e-6  --ltd $LTD --seed $SEED --wandb --save &