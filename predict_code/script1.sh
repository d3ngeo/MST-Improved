
#!/bin/bash
#!/usr/bin/env python3

fuser -v /dev/nvidia* | awk '{print $2}' | xargs -I{} kill -9 {}




# reconstruct by MST++
python3 test.py --rgb_path ./demo/ARAD_1K_0912.jpg --method mst_plus_plus --pretrained_model_path model_zoo-20250113T020157Z-001/model_zoo/mst_plus_plus.pth --outf ./exp/mst_plus_plus/ --gpu_id 0


