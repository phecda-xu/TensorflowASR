#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 \
python train_am.py \
--data_config configs/am_data_tdnn.yml \
--model_config  configs/tdnn.yml \
