import torch
import numpy as np
import argparse
import os
import torch.backends.cudnn as cudnn
from architecture import model_generator
from utils import AverageMeter, save_matv73, Loss_MRAE, Loss_RMSE, Loss_PSNR
from hsi_dataset import ValidDataset
from torch.utils.data import DataLoader
import warnings

# Suppress warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

# Argument parser
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--data_root', type=str, default='/media/ntu/volume1/home/s124md303_06/MST-plus-plus/dataset/')
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default='/media/ntu/volume1/home/s124md303_06/MST-plus-plus/model_zoo/mst_plus_plus.pth')
parser.add_argument('--outf', type=str, default='/media/ntu/volume1/home/s124md303_06/MST-plus-plus/exp/mst_plus_plus/')
parser.add_argument("--gpu_id", type=str, default='0')
opt = parser.parse_args()

# Set CUDA device
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# Debugging paths
print(f"Data root path: {opt.data_root}")
print(f"Pretrained model path: {opt.pretrained_model_path}")
print(f"Output directory: {opt.outf}")

# Create output directory if it doesn't exist
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)
    print(f"Created output directory: {opt.outf}")
else:
    print(f"Output directory already exists: {opt.outf}")

# Load dataset
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)

# Create DataLoader
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs).contiguous()
    targets = torch.stack(targets).contiguous()
    print(f"Collated batch - Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
    return inputs, targets

val_loader = DataLoader(dataset=val_data, batch_size=1, collate_fn=collate_fn, shuffle=False, num_workers=2, pin_memory=True)
print("DataLoader created successfully.")

# Loss functions
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
if torch.cuda.is_available():
    criterion_mrae.cuda()
    criterion_rmse.cuda()

# Validate
valid_list_path = os.path.join(opt.data_root, 'split_txt', 'valid_list.txt')
with open(valid_list_path, 'r') as fin:
    hyper_list = [line.strip() + '.mat' for line in fin]

# Debugging information
print(f"Loaded {len(hyper_list)} validation items from: {valid_list_path}")

hyper_list.sort()
var_name = 'cube'


def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()

    for i, (input, target) in enumerate(val_loader):
        # Debugging input and target
        print(f'Batch {i}: Input type {type(input)}, Input shape {input.shape}')
        print(f'Batch {i}: Target type {type(target)}, Target shape {target.shape}')

        try:
            # Ensure tensors are contiguous
            input = input.contiguous().cuda(non_blocking=True)
            target = target.contiguous().cuda(non_blocking=True)

            # Debugging strides
            print(f'Input strides: {input.stride()}, Target strides: {target.stride()}')

            with torch.no_grad():
                if opt.method == 'awan':
                    # Validate slicing requirements
                    if input.shape[2] <= 236 or input.shape[3] <= 236:
                        raise ValueError("Input height and width must be greater than 236 for slicing.")

                    output = model(input[:, :, 118:-118, 118:-118])

                    if output.shape[2] <= 20 or output.shape[3] <= 20 or target.shape[2] <= 256 or target.shape[3] <= 256:
                        raise ValueError("Output or target dimensions are too small for the specified slicing.")

                    output = output.contiguous()
                    loss_mrae = criterion_mrae(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
                    loss_rmse = criterion_rmse(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
                    loss_psnr = criterion_psnr(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
                else:
                    # Validate slicing requirements
                    if input.shape[2] <= 256 or input.shape[3] <= 256:
                        raise ValueError("Input height and width must be greater than 256 for slicing.")

                    output = model(input)

                    if output.shape[2] <= 256 or output.shape[3] <= 256 or target.shape[2] <= 256 or target.shape[3] <= 256:
                        raise ValueError("Output or target dimensions are too small for the specified slicing.")

                    output = output.contiguous()
                    loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
                    loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
                    loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])

            # Record loss
            losses_mrae.update(loss_mrae.item())
            losses_rmse.update(loss_rmse.item())
            losses_psnr.update(loss_psnr.item())

            # Process and save the output result
            result = output.cpu().numpy()
            result = np.transpose(np.squeeze(result), [1, 2, 0])
            result = np.clip(result, 0, 1)
            mat_name = hyper_list[i]
            mat_dir = os.path.join(opt.outf, mat_name)
            save_matv73(mat_dir, var_name, result)

        except Exception as e:
            print(f"Error processing batch {i}: {e}")

    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg


if __name__ == '__main__':
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_generator(method, pretrained_model_path).cuda()

    try:
        mrae, rmse, psnr = validate(val_loader, model)
        print(f'method: {method}, mrae: {mrae}, rmse: {rmse}, psnr: {psnr}')
    except Exception as e:
        print(f"Error during validation: {e}")
