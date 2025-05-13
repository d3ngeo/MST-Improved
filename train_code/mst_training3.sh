#!/bin/bash

# Ensure training continues even after logging off the GPU
LOG_FILE="training_log_mst_plus_plusx.log"
TENSORBOARD_BASE_DIR="./exp/mst_plus_plus"

# Remove old log file before starting a new one
rm -f $LOG_FILE

# Clear old TensorBoard logs (Optional but recommended)
rm -rf $TENSORBOARD_BASE_DIR/*/tensorboard_logs

echo "Starting MST++ training..."
nohup python3 train3.py \
  --method mst_plus_plus \
  --batch_size 10 \
  --end_epoch 100 \
  --init_lr 4e-4 \
  --outf $TENSORBOARD_BASE_DIR \
  --data_root ../dataset/ \
  --patch_size 128 \
  --stride 8 \
  --gpu_id 0 \
  > $LOG_FILE 2>&1 &

TRAIN_PID=$!
echo "Training process started with PID: $TRAIN_PID"

# Wait a short time to ensure logs start being written
sleep 10

# Find the latest log directory (most recent timestamped folder)
LATEST_LOG_DIR=$(ls -td $TENSORBOARD_BASE_DIR/* 2>/dev/null | head -n 1)/tensorboard_logs

# Kill any existing TensorBoard instance to prevent duplicates
pkill -f tensorboard 2>/dev/null

echo "Launching TensorBoard with logs from: $LATEST_LOG_DIR"
nohup tensorboard --logdir=$LATEST_LOG_DIR --port=6006 --bind_all > tensorboard.log 2>&1 &

TENSORBOARD_PID=$!
echo "TensorBoard launched with PID: $TENSORBOARD_PID"

# Disown the processes to fully detach from the shell
disown $TRAIN_PID
disown $TENSORBOARD_PID

echo "Training for MST++ has started and will continue even if you log off."
echo "Logs are being saved to $LOG_FILE"
echo "You can view training metrics at: http://localhost:6006"
