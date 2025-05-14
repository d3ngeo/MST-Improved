

#!/bin/bash
#!/usr/bin/env python3
# train MST-L
python3 train.py --method mst  --batch_size 20 --end_epoch 300 --init_lr 4e-4 --outf ./exp/mst/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0
