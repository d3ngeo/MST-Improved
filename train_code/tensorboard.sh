#!/bin/bash

echo "Starting tensorboard"
tensorboard --logdir=./exp/mst_plus_plus/2025_03_09_13_03_01/tensorboard_logs --port=6006 --bind_all
