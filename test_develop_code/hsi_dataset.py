from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import cv2
import h5py
import os
import torch

# Constants
MIN_DIM = 236  # Minimum dimension for valid inputs


class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8):
        # Paths and configurations
        data_root = '/media/ntu/volume1/home/s124md303_06/MST-plus-plus/dataset/'
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.arg = arg
        self.stride = stride

        # Compute number of patches per image
        h, w = 482, 512  # Default image dimensions
        self.patch_per_line = (w - crop_size) // stride + 1
        self.patch_per_column = (h - crop_size) // stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_column

        # Data paths
        hyper_data_path = os.path.join(data_root, 'Train_Spec/')
        bgr_data_path = os.path.join(data_root, 'Train_RGB/')

        # Load file lists
        with open(os.path.join(data_root, 'split_txt/train_list.txt'), 'r') as fin:
            file_list = [line.strip() for line in fin]
            hyper_list = [f"{file_name}.mat" for file_name in file_list]
            bgr_list = [f"{file_name}.jpg" for file_name in file_list]

        hyper_list.sort()
        bgr_list.sort()

        print(f"len(hyper) of ntire2022 dataset: {len(hyper_list)}")
        print(f"len(bgr) of ntire2022 dataset: {len(bgr_list)}")

        # Load hyperspectral and BGR images
        for i in range(len(hyper_list)):
            hyper_path = os.path.join(hyper_data_path, hyper_list[i])
            bgr_path = os.path.join(bgr_data_path, bgr_list[i])

            # Check file existence
            if not os.path.isfile(hyper_path):
                print(f"Skipping missing hyper file: {hyper_path}")
                continue
            if not os.path.isfile(bgr_path):
                print(f"Skipping missing BGR file: {bgr_path}")
                continue

            try:
                # Load hyperspectral data
                with h5py.File(hyper_path, 'r') as mat:
                    hyper = np.float32(mat['cube'][:])
                hyper = np.transpose(hyper, [0, 2, 1])  # Convert to [C, H, W]

                # Load BGR data
                bgr = cv2.imread(bgr_path)
                if bgr is None:
                    print(f"Failed to load BGR image: {bgr_path}")
                    continue
                if bgr2rgb:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bgr = np.float32(bgr) / 255.0  # Normalize to [0, 1]
                bgr = np.transpose(bgr, [2, 0, 1])  # Convert to [C, H, W]

                # Skip small dimensions
                if hyper.shape[1] < MIN_DIM or hyper.shape[2] < MIN_DIM or \
                   bgr.shape[1] < MIN_DIM or bgr.shape[2] < MIN_DIM:
                    print(f"Skipping file due to small dimensions: {hyper_path}")
                    continue

                # Append to dataset
                self.hypers.append(np.ascontiguousarray(hyper))
                self.bgrs.append(np.ascontiguousarray(bgr))

                print(f"Loaded scene {i}: Hyper: {hyper_path}, BGR: {bgr_path}")

            except Exception as e:
                print(f"Error processing files: Hyper: {hyper_path}, BGR: {bgr_path}, Error: {e}")

        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

        print(f"Total hyperspectral images loaded: {self.img_num}")

    def arguement(self, img, rotTimes, vFlip, hFlip):
        for _ in range(rotTimes):
            img = np.ascontiguousarray(np.rot90(img, axes=(1, 2)))
        if vFlip:
            img = np.ascontiguousarray(img[:, :, ::-1])
        if hFlip:
            img = np.ascontiguousarray(img[:, ::-1, :])
        return img

    def __getitem__(self, idx):
        img_idx, patch_idx = divmod(idx, self.patch_per_img)
        h_idx, w_idx = divmod(patch_idx, self.patch_per_line)

        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]

        crop_h = slice(h_idx * self.stride, h_idx * self.stride + self.crop_size)
        crop_w = slice(w_idx * self.stride, w_idx * self.stride + self.crop_size)

        bgr = torch.from_numpy(np.ascontiguousarray(bgr[:, crop_h, crop_w])).float()
        hyper = torch.from_numpy(np.ascontiguousarray(hyper[:, crop_h, crop_w])).float()

        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            bgr = torch.from_numpy(self.arguement(bgr.numpy(), rotTimes, vFlip, hFlip)).float()
            hyper = torch.from_numpy(self.arguement(hyper.numpy(), rotTimes, vFlip, hFlip)).float()

        return bgr, hyper

    def __len__(self):
        return self.patch_per_img * self.img_num


class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        data_root = '/media/ntu/volume1/home/s124md303_06/MST-plus-plus/dataset/'
        self.hypers = []
        self.bgrs = []

        # Data paths
        hyper_data_path = os.path.join(data_root, 'Train_Spec/')
        bgr_data_path = os.path.join(data_root, 'Train_RGB/')

        # Load file lists
        with open(os.path.join(data_root, 'split_txt/valid_list.txt'), 'r') as fin:
            file_list = [line.strip() for line in fin]
            hyper_list = [f"{file_name}.mat" for file_name in file_list]
            bgr_list = [f"{file_name}.jpg" for file_name in file_list]

        hyper_list.sort()
        bgr_list.sort()

        print(f"len(hyper_valid) of ntire2022 dataset: {len(hyper_list)}")
        print(f"len(bgr_valid) of ntire2022 dataset: {len(bgr_list)}")

        # Load hyperspectral and BGR images
        for i in range(len(hyper_list)):
            hyper_path = os.path.join(hyper_data_path, hyper_list[i])
            bgr_path = os.path.join(bgr_data_path, bgr_list[i])

            if not os.path.isfile(hyper_path):
                print(f"Skipping missing hyper file: {hyper_path}")
                continue
            if not os.path.isfile(bgr_path):
                print(f"Skipping missing BGR file: {bgr_path}")
                continue

            try:
                # Load hyperspectral data
                with h5py.File(hyper_path, 'r') as mat:
                    hyper = np.float32(mat['cube'][:])
                hyper = np.transpose(hyper, [0, 2, 1])

                # Load BGR data
                bgr = cv2.imread(bgr_path)
                if bgr is None:
                    print(f"Failed to load BGR image: {bgr_path}")
                    continue
                if bgr2rgb:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bgr = np.float32(bgr) / 255.0
                bgr = np.transpose(bgr, [2, 0, 1])

                # Append to dataset
                self.hypers.append(np.ascontiguousarray(hyper))
                self.bgrs.append(np.ascontiguousarray(bgr))

            except Exception as e:
                print(f"Error processing files: Hyper: {hyper_path}, BGR: {bgr_path}, Error: {e}")

    def __getitem__(self, idx):
        bgr = torch.from_numpy(np.ascontiguousarray(self.bgrs[idx])).float()
        hyper = torch.from_numpy(np.ascontiguousarray(self.hypers[idx])).float()
        return bgr, hyper

    def __len__(self):
        return len(self.hypers)
