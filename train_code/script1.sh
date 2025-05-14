
#!/usr/bin/env python3



# train MST++
python3 train.py --method mst_plus_plus  --batch_size 20 --end_epoch 300 --init_lr 4e-4 --outf ./exp/mst_plus_plus/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0

# train MST-L
python3 train.py --method mst  --batch_size 20 --end_epoch 300 --init_lr 4e-4 --outf ./exp/mst/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0

# train MIRNet
python3 train.py --method mirnet  --batch_size 20 --end_epoch 300 --init_lr 4e-4 --outf ./exp/mirnet/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0

# train HINet
python3 train.py --method hinet  --batch_size 20 --end_epoch 300 --init_lr 2e-4 --outf ./exp/hinet/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0

# train MPRNet
python3 train.py --method mprnet  --batch_size 20 --end_epoch 300 --init_lr 2e-4 --outf ./exp/mprnet/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0

# train Restormer
python3 train.py --method restormer  --batch_size 20 --end_epoch 300 --init_lr 2e-4 --outf ./exp/restormer/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0

# train EDSR
python3 train.py --method edsr  --batch_size 20 --end_epoch 300 --init_lr 1e-4 --outf ./exp/edsr/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0

# train HDNet
python3 train.py --method hdnet  --batch_size 20 --end_epoch 300 --init_lr 4e-4 --outf ./exp/hdnet/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0

# train HRNet
python3 train.py --method hrnet  --batch_size 20 --end_epoch 300 --init_lr 1e-4 --outf ./exp/hrnet/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0

# train HSCNN+
python3 train.py --method hscnn_plus  --batch_size 20 --end_epoch 300 --init_lr 2e-4 --outf ./exp/hscnn_plus/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0

# train AWAN
python3 train.py --method awan  --batch_size 20 --end_epoch 300 --init_lr 1e-4 --outf ./exp/awan/ --data_root ../dataset/  --patch_size 128 --stride 8  --gpu_id 0