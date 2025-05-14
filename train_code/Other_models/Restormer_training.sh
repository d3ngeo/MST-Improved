#!/bin/bash

# Ensure training continues even after logging off the GPU
LOG_FILE="training_log_restormer_$(date +'%Y%m%d_%H%M%S').log"

nohup python3 train.py \
  --method restormer \
  --batch_size 10 \
  --end_epoch 300 \
  --init_lr 2e-4 \
  --outf ./exp/restormer/ \
  --data_root ../dataset/ \
  --patch_size 128 \
  --stride 8 \
  --gpu_id 0 \
  | tee -a $LOG_FILE &

echo "Training for Restormer has started and will continue even if you log off. Logs are being saved to $LOG_FILE and printed to the console."
