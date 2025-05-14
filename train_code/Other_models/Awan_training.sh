#!/bin/bash
#!/usr/bin/env python3


# train AWAN
python3 train.py --method awan  --batch_size 20 --end_epoch 300 --init_lr 1e-4 --outf ./exp/awan/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0
