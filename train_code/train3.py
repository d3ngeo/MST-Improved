import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from hsi_dataset import TrainDataset, ValidDataset
from architecture import *
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR
import datetime
from losses import total_variation_loss, ssim_loss
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=10, help="batch size")
parser.add_argument("--end_epoch", type=int, default=300, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=6e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='./exp/mst_plus_plus/', help='path log files')
parser.add_argument("--data_root", type=str, default='../dataset/')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='GPU ID')
opt = parser.parse_args()

# GPU Configuration
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
cudnn.benchmark = True
cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Load dataset
print("\nLoading dataset...")
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=opt.stride)
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)

# Define iteration limit
per_epoch_iteration = 1000
total_iterations = per_epoch_iteration * opt.end_epoch

# Loss functions
crion_mrae = Loss_MRAE()
crion_rmse = Loss_RMSE()
crion_psnr = Loss_PSNR()

# Model
model = model_generator(opt.method, opt.pretrained_model_path).cuda()
print('Parameters number is', sum(param.numel() for param in model.parameters()))

# Output path setup
date_time = time2file_name(str(datetime.datetime.now()))
opt.outf = os.path.join(opt.outf, date_time)
os.makedirs(opt.outf, exist_ok=True)

# TensorBoard setup
tensorboard_log_dir = os.path.join(opt.outf, "tensorboard_logs")
os.makedirs(tensorboard_log_dir, exist_ok=True)
writer = SummaryWriter(tensorboard_log_dir)

# ?? Immediately log a dummy scalar to force TensorBoard log file creation
writer.add_scalar("Debug/Startup", 0, 0)
writer.flush()  # Ensure it is written

# Move model and loss functions to GPU
if torch.cuda.is_available():
    model.cuda()
    crion_mrae.cuda()
    crion_rmse.cuda()
    crion_psnr.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iterations, eta_min=1e-6)

# Logging
log_dir = os.path.join(opt.outf, 'train.log')
logger = initialize_logger(log_dir)

# Resume from checkpoint
iteration = 0
resume_file = opt.pretrained_model_path
if resume_file is not None and os.path.isfile(resume_file):
    print("=> Loading checkpoint '{}'".format(resume_file))
    checkpoint = torch.load(resume_file)
    iteration = checkpoint.get('iteration', 0)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

scaler = GradScaler()

def main():
    global iteration
    model.train()
    
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    while iteration < total_iterations:
        for _, (images, labels) in enumerate(train_loader):
            if iteration >= total_iterations:
                break

            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            output = model(images)
            loss = crion_mrae(output, labels) + 0.1 * ssim_loss(output, labels, window_size=5) + 1e-8 * total_variation_loss(output)
            loss.backward()
            optimizer.step()
            scheduler.step()

            iteration += 1

            # Log training loss every 20 iterations
            if iteration % 20 == 0:
                writer.add_scalar("Loss/Train", loss.item(), iteration)
                writer.add_scalar("LR", optimizer.param_groups[0]["lr"], iteration)
                writer.flush()  # Force writing to TensorBoard
                print(f'Iteration [{iteration}/{total_iterations}], LR: {optimizer.param_groups[0]["lr"]:.9f}, Loss: {loss.item():.9f}')

            # Perform validation every 1000 iterations
            if iteration % 1000 == 0:
                validate(val_loader, model)
                save_checkpoint(opt.outf, iteration // 1000, iteration, model, optimizer)

    print("Training complete.")
    writer.close()

def validate(val_loader, model):
    global iteration  # Ensures TensorBoard can use the current iteration value
    model.eval()
    losses_mrae, losses_rmse, losses_psnr = AverageMeter(), AverageMeter(), AverageMeter()

    with torch.no_grad():
        for input, target in val_loader:
            input, target = input.cuda(), target.cuda()
            target = target + 1e-6  # Prevent zero target issues
            output = model(input)

            losses_mrae.update(crion_mrae(output, target).item())
            losses_rmse.update(crion_rmse(output, target).item())
            losses_psnr.update(crion_psnr(output, target).item())

    writer.add_scalar("Validation/MRAE", losses_mrae.avg, iteration)
    writer.add_scalar("Validation/RMSE", losses_rmse.avg, iteration)
    writer.add_scalar("Validation/PSNR", losses_psnr.avg, iteration)
    writer.flush()  # Ensure logs are written immediately

    print(f'Validation - MRAE: {losses_mrae.avg:.6f}, RMSE: {losses_rmse.avg:.6f}, PSNR: {losses_psnr.avg:.6f}')

if __name__ == '__main__':
    main()


