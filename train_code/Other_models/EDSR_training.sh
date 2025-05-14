#!/bin/bash
#!/usr/bin/env python3

# train EDSR
python3 train.py --method edsr  --batch_size 20 --end_epoch 300 --init_lr 1e-4 --outf ./exp/edsr/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0
