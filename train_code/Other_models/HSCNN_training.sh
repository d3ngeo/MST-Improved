#!/bin/bash

#!/usr/bin/env python3

# train HSCNN+
python3 train.py --method hscnn_plus  --batch_size 20 --end_epoch 300 --init_lr 2e-4 --outf ./exp/hscnn_plus/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0


