#!/bin/bash
#!/usr/bin/env python3

fuser -v /dev/nvidia* | awk '{print $2}' | xargs -I{} kill -9 {}


# train MST++
python3 train.py --method mst_plus_plus  --batch_size 20 --end_epoch 300 --init_lr 4e-4 --outf ./exp/mst_plus_plus/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0
