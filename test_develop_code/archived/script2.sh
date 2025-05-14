#!/bin/bash 
fuser -v /dev/nvidia* | awk '{print $2}' | xargs -I{} kill -9 {}

# Define the path to the pretrained model files
pretrained_model_path="/media/ntu/volume1/home/s124md303_06/MST-plus-plus/test_develop_code/model_zoo/model_zoo"

# Check if the pretrained model path exists
if [ ! -d "$pretrained_model_path" ]; then
    echo "Error: Pretrained model path does not exist: $pretrained_model_path"
    exit 1
fi

# Define the data root path
data_root="/media/ntu/volume1/home/s124md303_06/MST-plus-plus/dataset/"

# Check if the data root path exists
if [ ! -d "$data_root" ]; then
    echo "Error: Data root path does not exist: $data_root"
    exit 1
fi

# test MST++
python3 test2.py --data_root "$data_root" --method mst_plus_plus --pretrained_model_path "${pretrained_model_path}/mst_plus_plus.pth" --outf ./exp/mst_plus_plus/ --gpu_id 0

# test MST-L
python3 test2.py --data_root "$data_root" --method mst --pretrained_model_path "${pretrained_model_path}/mst.pth" --outf ./exp/mst/ --gpu_id 0

# test MIRNet
python3 test2.py --data_root "$data_root" --method mirnet --pretrained_model_path "${pretrained_model_path}/mirnet.pth" --outf ./exp/mirnet/ --gpu_id 0

# test HINet
python3 test2.py --data_root "$data_root" --method hinet --pretrained_model_path "${pretrained_model_path}/hinet.pth" --outf ./exp/hinet/ --gpu_id 0

# test MPRNet
python3 test2.py --data_root "$data_root" --method mprnet --pretrained_model_path "${pretrained_model_path}/mprnet.pth" --outf ./exp/mprnet/ --gpu_id 0

# test Restormer
python3 test2.py --data_root "$data_root" --method restormer --pretrained_model_path "${pretrained_model_path}/restormer.pth" --outf ./exp/restormer/ --gpu_id 0

# test EDSR
python3 test2.py --data_root "$data_root" --method edsr --pretrained_model_path "${pretrained_model_path}/edsr.pth" --outf ./exp/edsr/ --gpu_id 0

# test HDNet
python3 test2.py --data_root "$data_root" --method hdnet --pretrained_model_path "${pretrained_model_path}/hdnet.pth" --outf ./exp/hdnet/ --gpu_id 0

# test HRNet
python3 test2.py --data_root "$data_root" --method hrnet --pretrained_model_path "${pretrained_model_path}/hrnet.pth" --outf ./exp/hrnet/ --gpu_id 0

# test HSCNN+
python3 test2.py --data_root "$data_root" --method hscnn_plus --pretrained_model_path "${pretrained_model_path}/hscnn_plus.pth" --outf ./exp/hscnn_plus/ --gpu_id 0

# test AWAN
python3 test2.py --data_root "$data_root" --method awan --pretrained_model_path "${pretrained_model_path}/awan.pth" --outf ./exp/awan/ --gpu_id 0

echo "All tests completed."

