#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

WORKSPACE='/path/to/checkpoints/0801/'
VERSION=example

DEFAULT_PARAMS=./configs/example.yaml

python train.py \
    --config_path $DEFAULT_PARAMS \
    --workspace $WORKSPACE --version $VERSION \
    --extra_config '{}'