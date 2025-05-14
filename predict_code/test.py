import torch
import argparse
import torch.backends.cudnn as cudnn
import os
from architecture import *
from utils import save_matv73
import cv2
import numpy as np
import itertools

parser = argparse.ArgumentParser(description="SSR")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default='./model_zoo/mst_plus_plus.pth')
parser.add_argument('--rgb_path', type=str, default='./demo/ARAD_1K_0912.jpg')
parser.add_argument('--outf', type=str, default='./exp/mst_plus_plus/')
parser.add_argument('--ensemble_mode', type=str, default='mean')
parser.add_argument("--gpu_id", type=str, default='0')
opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

def main():
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method

    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model to appropriate device
    model = model_generator(method, pretrained_model_path).to(device)
    test(model, opt.rgb_path, opt.outf, device)

def test(model, rgb_path, save_path, device):
    var_name = 'cube'
    bgr = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Resize the input image to reduce memory usage
    rgb = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2))
    rgb = np.float32(rgb)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0).copy()
    rgb = torch.from_numpy(rgb).float().to(device)

    print(f'Reconstructing {rgb_path}')
    with torch.no_grad():
        # Use mixed precision to save memory
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            result = forward_ensemble(rgb, model, opt.ensemble_mode)

    result = result.cpu().numpy() * 1.0
    result = np.transpose(np.squeeze(result), [1, 2, 0])
    result = np.minimum(result, 1.0)
    result = np.maximum(result, 0)

    # Convert result to float32
    result = result.astype(np.float32)

    mat_name = rgb_path.split('/')[-1][:-4] + '.mat'
    mat_dir = os.path.join(save_path, mat_name)
    save_matv73(mat_dir, var_name, result)
    print(f'The reconstructed hyper spectral image is saved as {mat_dir}.')

def forward_ensemble(x, forward_func, ensemble_mode='mean'):
    def _transform(data, xflip, yflip, transpose, reverse=False):
        if not reverse:  # forward transform
            if xflip:
                data = torch.flip(data, [3])
            if yflip:
                data = torch.flip(data, [2])
            if transpose:
                data = torch.transpose(data, 2, 3)
        else:  # reverse transform
            if transpose:
                data = torch.transpose(data, 2, 3)
            if yflip:
                data = torch.flip(data, [2])
            if xflip:
                data = torch.flip(data, [3])
        return data

    outputs = []
    opts = itertools.product((False, True), (False, True), (False, True))
    for xflip, yflip, transpose in opts:
        data = x.clone()
        data = _transform(data, xflip, yflip, transpose)
        data = forward_func(data)
        outputs.append(
            _transform(data, xflip, yflip, transpose, reverse=True))
    if ensemble_mode == 'mean':
        return torch.stack(outputs, 0).mean(0)
    elif ensemble_mode == 'median':
        return torch.stack(outputs, 0).median(0)[0]

if __name__ == '__main__':
    torch.cuda.empty_cache()  # Clear GPU memory before starting
    main()
