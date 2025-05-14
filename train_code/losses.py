import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================
# Total Variation (TV) Loss
# ============================
def total_variation_loss(img):
    """
    Compute the Total Variation (TV) Loss to encourage spatial smoothness.
    Args:
        img (torch.Tensor): Image tensor of shape (B, C, H, W)
    Returns:
        torch.Tensor: Total Variation loss value
    """
    loss_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    loss_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return (loss_h + loss_w).clamp(min=1e-6)  # Prevent zero loss issues


# ============================
# Structural Similarity Index (SSIM) Loss
# ============================
def ssim_loss(img1, img2, window_size=5):
    """
    Compute the SSIM loss to preserve perceptual quality.
    Args:
        img1 (torch.Tensor): Predicted output
        img2 (torch.Tensor): Ground truth
    Returns:
        torch.Tensor: SSIM-based loss value
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

    sigma1 = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size // 2) - mu1 ** 2
    sigma2 = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size // 2) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1 * mu2

    sigma1 = sigma1.clamp(min=1e-6)
    sigma2 = sigma2.clamp(min=1e-6)
    sigma12 = sigma12.clamp(min=1e-6)

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return 1 - ssim_map.mean()  # SSIM loss (1 - SSIM score)


# ============================
# Hybrid Loss (MRAE + SSIM + TV)
# ============================
class HybridLoss(nn.Module):
    def __init__(self, ssim_weight=0.1, tv_weight=0.00001):
        """
        Hybrid loss function combining:
        - MRAE (Mean Relative Absolute Error)
        - SSIM Loss (Structural Similarity)
        - TV Loss (Total Variation)

        Args:
            ssim_weight (float): Weight for SSIM loss
            tv_weight (float): Weight for TV loss
        """
        super(HybridLoss, self).__init__()
        self.ssim_weight = ssim_weight
        self.tv_weight = tv_weight

    def forward(self, output, target):
        """
        Compute the hybrid loss.
        Args:
            output (torch.Tensor): Predicted output
            target (torch.Tensor): Ground truth
        Returns:
            torch.Tensor: Hybrid loss value
        """
        epsilon = 1e-6  # To avoid division by zero in MRAE
        mrae_loss = torch.mean(torch.log(1 + torch.abs(output - target) / (torch.abs(target) + epsilon)))  # More stable
        ssim_loss_value = ssim_loss(output, target)  # SSIM Loss
        tv_loss_value = total_variation_loss(output)  # TV Loss

        hybrid_loss = mrae_loss + self.ssim_weight * ssim_loss_value + self.tv_weight * tv_loss_value
        return hybrid_loss

