

#!/bin/bash

# Ensure training continues even after logging off the GPU
LOG_FILE="training_log_mst_plus_plus.log"

# Remove old log file before starting a new one
rm -f $LOG_FILE

nohup python3 train2.py \
  --method mst_plus_plus \
  --batch_size 20 \
  --end_epoch 300 \
  --init_lr 4e-4 \
  --outf ./exp/mst_plus_plus/ \
  --data_root ../dataset/ \
  --patch_size 128 \
  --stride 8 \
  --gpu_id 0 \
  | tee $LOG_FILE &

echo "Training for MST++ has started and will continue even if you log off. Logs are being saved to $LOG_FILE and printed to the console."
