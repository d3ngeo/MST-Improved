#!/bin/bash
#!/usr/bin/env python3

# train HDNet
python3 train.py --method hdnet  --batch_size 20 --end_epoch 300 --init_lr 4e-4 --outf ./exp/hdnet/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0
