#!/bin/bash

python ./src/train_ssl.py \
    --debug \
    --datapath data// \
    --seed 10 \
    --dataset cora \
    --type mutigcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 0 \
    --hidden 128 \
    --epoch 200 \
    --lr 0.01 \
    --weight_decay 5e-04 \
    --early_stopping 200 \
    --sampling_percent 1 \
    --dropout 0.5 \
    --normalization AugNormAdj --task_type semi \
    --ssl ICAContextLabel \
    --lambda_ 5 \
    --train_size 10 \
    --param_searching 0 \
    --alpha 2
     \


