#!/usr/bin/env bash

SEED=0
EPOCH=100
ENV=mnist1
# wide
python -m main --epoch $EPOCH --env $ENV --granule fc --k 4    --n-hidden 5000     --lr 1e-4     --ltd none  --seed $SEED --wandb --save &
python -m main --epoch $EPOCH --env $ENV --granule fc --k 4    --n-hidden 50000    --lr 1e-5     --ltd none  --seed $SEED --wandb --save &
python -m main --epoch $EPOCH --env $ENV --granule fc --k 4    --n-hidden 200000   --lr 2.5e-6   --ltd none  --seed $SEED --wandb --save &
# ltd
python -m main --epoch $EPOCH --env $ENV --granule fc --k 4    --n-hidden 5000     --lr 1e-4     --ltd ma    --seed $SEED --wandb --save &
# sparse
python -m main --epoch $EPOCH --env $ENV --granule rc --k 4    --n-hidden 5000     --lr 1e-4     --ltd ma    --seed $SEED --wandb --save &
python -m main --epoch $EPOCH --env $ENV --granule rc --k 20   --n-hidden 5000     --lr 1e-4     --ltd ma    --seed $SEED --wandb --save &
python -m main --epoch $EPOCH --env $ENV --granule rc --k 100  --n-hidden 5000     --lr 1e-4     --ltd ma    --seed $SEED --wandb --save &
# sparse
python -m main --epoch $EPOCH --env $ENV --granule rc --k 4    --n-hidden 200000   --lr 2.5e-6   --ltd ma    --seed $SEED --wandb --save &
python -m main --epoch $EPOCH --env $ENV --granule rc --k 20   --n-hidden 120000   --lr 4.167e-6 --ltd ma    --seed $SEED --wandb --save &
python -m main --epoch $EPOCH --env $ENV --granule rc --k 100  --n-hidden 40000    --lr 1.25e-5  --ltd ma    --seed $SEED --wandb --save &