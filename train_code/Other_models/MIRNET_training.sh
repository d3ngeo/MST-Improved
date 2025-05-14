#!/bin/bash


#!/usr/bin/env python3

# train MIRNet
python3 train.py --method mirnet  --batch_size 20 --end_epoch 300 --init_lr 4e-4 --outf ./exp/mirnet/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0
