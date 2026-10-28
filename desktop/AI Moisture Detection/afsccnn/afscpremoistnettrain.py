"""
Comprehensive AFSC (DCT) + ResNet integration script.

Features:
- AFSC_DCT: Adaptive Frequency-Selective Convolution using 2-D DCT (orthonormal DCT-II).
- Low-frequency block mask (learnable): small mask_param applied to top-left low-frequency region.
- TensorBoard logging: mask heatmaps per epoch, training/validation metrics.
- Utilities to replace multiple ResNet blocks (gradual replacement).
- Warm-start from pretrained conv weights, mask regularization, param groups, mixed precision.

Author: Arham Sethi
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import datetime
import torchvision
from torchvision.models import resnet18, ResNet18_Weights  # Added for deprecated fix
import csv

# ------------------------------
# Utilities: DCT basis matrix
# ------------------------------
def create_dct_matrix(N, device=None, dtype=torch.float32):
    if device is None:
        device = torch.device('cpu')
    k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)
    n = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)
    alpha = torch.ones(N, device=device, dtype=dtype) * math.sqrt(2.0 / N)
    alpha[0] = math.sqrt(1.0 / N)
    angles = math.pi * (n + 0.5) * k / N
    D = alpha.unsqueeze(1) * torch.cos(angles)
    return D


# ------------------------------
# AFSC DCT Layer
# ------------------------------
class AFSC_DCT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True,
                 input_size=(224, 224), low_freq_size=(16, 16),
                 mask_mode='per_out', device=None, dtype=torch.float32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)
        self.input_size = tuple(input_size)
        self.low_freq_size = tuple(low_freq_size)
        assert mask_mode in ('per_out', 'global')
        self.mask_mode = mask_mode
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels,
                                               self.kernel_size[0], self.kernel_size[1]) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Use padded size for DCT matrices
        H, W = self.input_size
        padded_H = H + 2 * self.padding[0]
        padded_W = W + 2 * self.padding[1]
        D_H = create_dct_matrix(padded_H, device=self.device, dtype=self.dtype)
        D_W = create_dct_matrix(padded_W, device=self.device, dtype=self.dtype)
        self.register_buffer('D_H', D_H)
        self.register_buffer('D_W', D_W)
        self.register_buffer('D_W_T', D_W.t())

        lf_h, lf_w = self.low_freq_size
        if self.mask_mode == 'per_out':
            mask_shape = (out_channels, lf_h, lf_w)
        else:
            mask_shape = (1, lf_h, lf_w)
        init_val = 1.5
        self.mask_param = nn.Parameter(torch.full(mask_shape, init_val,
                                                  dtype=self.dtype, device=self.device))

    def dct2(self, x):
        B, C, H, W = x.shape
        assert H == self.input_size[0] + 2 * self.padding[0], f"Expected height {self.input_size[0] + 2 * self.padding[0]}, got {H}"
        assert W == self.input_size[1] + 2 * self.padding[1], f"Expected width {self.input_size[1] + 2 * self.padding[1]}, got {W}"
        x_bc = x.view(B * C, H, W)
        tmp = torch.matmul(self.D_H, x_bc)
        X = torch.matmul(tmp, self.D_W_T)
        return X.view(B, C, H, W)

    def idct2(self, X):
        B, C, H, W = X.shape
        X_bc = X.view(B * C, H, W)
        tmp = torch.matmul(self.D_H.t(), X_bc)
        x = torch.matmul(tmp, self.D_W)
        return x.view(B, C, H, W)

    def forward(self, x):
        B, C_in, H, W = x.shape
        print(f"AFSC_DCT input shape: {x.shape}, Expected: {self.input_size}")  # Debug
        assert (H, W) == self.input_size, f"Expected input size {self.input_size}, got {(H, W)}"
        if self.padding != (0, 0):
            x = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]), mode='constant', value=0)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN or Inf detected in input")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        X_freq = self.dct2(x)
        kH, kW = self.kernel_size
        padded_H = H + 2 * self.padding[0]
        padded_W = W + 2 * self.padding[1]
        kernel_pad = torch.zeros((self.out_channels, self.in_channels, padded_H, padded_W),
                                 device=self.weight.device, dtype=self.weight.dtype)
        kernel_pad[:, :, :kH, :kW] = self.weight
        shift_h, shift_w = -(kH // 2), -(kW // 2)
        kernel_pad = torch.roll(kernel_pad, shifts=(shift_h, shift_w), dims=(2, 3))

        kernel_bc = kernel_pad.view(self.out_channels * self.in_channels, padded_H, padded_W)
        tmp_k = torch.matmul(self.D_H, kernel_bc)
        K_freq_all = torch.matmul(tmp_k, self.D_W_T)
        K_freq = K_freq_all.view(self.out_channels, self.in_channels, padded_H, padded_W)

        Xf = X_freq.unsqueeze(1)
        Kf = K_freq.unsqueeze(0)
        out_freq = torch.sum(Xf * Kf, dim=2)

        lf_h, lf_w = self.low_freq_size
        if self.mask_mode == 'per_out':
            mask_small = torch.sigmoid(self.mask_param)
            mask_full = torch.zeros((self.out_channels, padded_H, padded_W), device=mask_small.device)
            mask_full[:, :lf_h, :lf_w] = mask_small
        else:
            mask_small = torch.sigmoid(self.mask_param)
            mask_full = torch.zeros((self.out_channels, padded_H, padded_W), device=mask_small.device)
            mask_full[:, :lf_h, :lf_w] = mask_small[0].unsqueeze(0)

        mask_full_b = mask_full.unsqueeze(0).expand(B, -1, -1, -1)
        out_freq_masked = out_freq * mask_full_b
        out_spatial = self.idct2(out_freq_masked)

        # Crop to match standard convolution output before applying stride
        out_spatial = out_spatial[:, :, self.padding[0]:self.padding[0] + H, self.padding[1]:self.padding[1] + W]

        if any(s > 1 for s in self.stride):
            out_spatial = out_spatial[:, :, ::self.stride[0], ::self.stride[1]]

        if self.bias is not None:
            out_spatial = out_spatial + self.bias.view(1, -1, 1, 1)

        return out_spatial

    def get_mask_for_visualization(self):
        with torch.no_grad():
            lf_h, lf_w = self.low_freq_size
            H, W = self.input_size
            padded_H = H + 2 * self.padding[0]
            padded_W = W + 2 * self.padding[1]
            if self.mask_mode == 'per_out':
                mask_small = torch.sigmoid(self.mask_param).detach().cpu()
                mask_full = torch.zeros((self.out_channels, padded_H, padded_W))
                mask_full[:, :lf_h, :lf_w] = mask_small
            else:
                mask_small = torch.sigmoid(self.mask_param).detach().cpu()
                mask_full = torch.zeros((self.out_channels, padded_H, padded_W))
                mask_full[:, :lf_h, :lf_w] = mask_small[0].unsqueeze(0)
        return mask_full


# ------------------------------
# Dataset
# ------------------------------
class MoistureDataset(Dataset):
    def __init__(self, csv_path, augment=False, resize=(224, 224)):
        self.data = pd.read_csv(csv_path)
        self.augment = augment
        base_transforms = [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        if self.augment:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image']
        label = float(row['folder'])
        label = torch.tensor(label, dtype=torch.float32)
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a zero tensor as fallback
            img = torch.zeros((3, 224, 224))
        return img, label


# ------------------------------
# Replace ResNet convs with AFSC
# ------------------------------
def replace_resnet_convs_with_afsc(model, input_size=(224, 224), low_freq_size=(16, 16),
                                   layers_to_replace=('conv1',), mask_mode='per_out', device=None):
    if device is None:
        device = torch.device('cpu')
    current_size = list(input_size)  # Start with initial input size

    if 'conv1' in layers_to_replace:
        old_conv = model.conv1
        afsc = AFSC_DCT(old_conv.in_channels, old_conv.out_channels,
                        old_conv.kernel_size, old_conv.stride, old_conv.padding,
                        bias=(old_conv.bias is not None), input_size=tuple(current_size),
                        low_freq_size=low_freq_size, mask_mode=mask_mode,
                        device=device, dtype=old_conv.weight.dtype)
        with torch.no_grad():
            afsc.weight.data.copy_(old_conv.weight.data)
            if old_conv.bias is not None:
                afsc.bias.data.copy_(old_conv.bias.data)
        model.conv1 = afsc
        # Update current_size after conv1
        current_size = [((s + 2 * old_conv.padding[0] - old_conv.kernel_size[0]) // old_conv.stride[0] + 1) for s in current_size]
        # Account for maxpool after conv1 (kernel=3, stride=2, padding=1)
        pool_padding = 1
        pool_kernel = 3
        pool_stride = 2
        current_size = [((s + 2 * pool_padding - pool_kernel) // pool_stride + 1) for s in current_size]

    for layer_name in ('layer1', 'layer2', 'layer3', 'layer4'):
        if layer_name in layers_to_replace:
            layer = getattr(model, layer_name)
            for block in layer:
                if hasattr(block, 'conv1'):
                    old_conv = block.conv1
                    afsc = AFSC_DCT(old_conv.in_channels, old_conv.out_channels,
                                    old_conv.kernel_size, old_conv.stride, old_conv.padding,
                                    bias=(old_conv.bias is not None), input_size=tuple(current_size),
                                    low_freq_size=low_freq_size, mask_mode=mask_mode,
                                    device=device, dtype=old_conv.weight.dtype)
                    with torch.no_grad():
                        afsc.weight.data.copy_(old_conv.weight.data)
                        if old_conv.bias is not None:
                            afsc.bias.data.copy_(old_conv.bias.data)
                    block.conv1 = afsc
                # Update current_size after this conv1
                current_size = [((s + 2 * old_conv.padding[0] - old_conv.kernel_size[0]) // old_conv.stride[0] + 1) for s in current_size]
    return model


# ------------------------------
# TensorBoard mask logging
# ------------------------------
def log_masks_to_tensorboard(model, writer, epoch, max_channels=8, prefix='AFSC'):
    for name, module in model.named_modules():
        if isinstance(module, AFSC_DCT):
            mask_full = module.get_mask_for_visualization()
            out_ch = mask_full.shape[0]
            n_plot = min(max_channels, out_ch)
            fig, axs = plt.subplots(1, n_plot, figsize=(3*n_plot, 3))
            if n_plot == 1:
                axs = [axs]
            for i in range(n_plot):
                ax = axs[i]
                m = mask_full[i].numpy()
                im = ax.imshow(m, origin='upper', cmap='viridis', aspect='auto')
                ax.set_title(f'{name} out#{i}')
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f'{prefix} mask heatmaps — {name} — epoch {epoch}')
            writer.add_figure(f'masks/{name}', fig, global_step=epoch)
            plt.close(fig)


# ------------------------------
# Updated Training loop
# ------------------------------
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import mean_squared_error
from torch.nn import SmoothL1Loss

def train_main(train_csv='train_clean.csv', val_csv='val_clean.csv', test_csv='test_clean.csv',
               replace_layers=('conv1',), low_freq_size=(16, 16),
               mask_mode='per_out', input_size=(224,224),
               batch_size=8, num_epochs=40, device=None, logdir='runs/afsc_dct_experiment'):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = MoistureDataset(train_csv, augment=True, resize=input_size)
    val_ds = MoistureDataset(val_csv, augment=False, resize=input_size)
    test_ds = MoistureDataset(test_csv, augment=False, resize=input_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = replace_resnet_convs_with_afsc(model, input_size=input_size, low_freq_size=low_freq_size,
                                           layers_to_replace=replace_layers, mask_mode=mask_mode, device=device)
    # Add dropout to prevent overfitting
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 1)
    )
    model = model.to(device)

    # Use more robust loss
    criterion = SmoothL1Loss()

    mask_params = [p for n, p in model.named_parameters() if 'mask_param' in n]
    other_params = [p for n, p in model.named_parameters() if 'mask_param' not in n]
    optimizer = torch.optim.AdamW([
        {'params': other_params},
        {'params': mask_params, 'lr': 5e-4}
    ], lr=1e-4, weight_decay=1e-5)

    # OneCycle learning rate schedule
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader),
                           epochs=num_epochs, div_factor=10, pct_start=0.3)

    scaler = torch.amp.GradScaler()
    writer = SummaryWriter(log_dir=os.path.join(logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    early_stop_counter = 0
    best_val_loss = float('inf')
    patience = 10
    mask_reg_base = 5e-4
    accumulation_steps = max(1, 16 // batch_size)
    warm_epochs = 3
    total_epochs = num_epochs

    # ------------------------------
    for name, p in model.named_parameters():
        p.requires_grad = ('mask_param' in name) or ('fc' in name)
    print(f"[INFO] Warming up for {warm_epochs} epochs...")

    for epoch in range(1, total_epochs + 1):
        model.train()
        if epoch == warm_epochs + 1:
            for p in model.parameters():
                p.requires_grad = True
            print("[INFO] Unfroze all parameters after warmup.")

        epoch_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            with torch.amp.autocast(device_type=device.type):
                preds = model(imgs)
                loss = criterion(preds, labels)

                # Mask regularization with cosine ramp-up
                progress = epoch / total_epochs
                mask_reg_weight = mask_reg_base * 0.5 * (1 - math.cos(math.pi * progress))
                mask_penalty = sum(torch.mean(torch.abs(torch.sigmoid(m.mask_param)))
                                   for m in model.modules() if isinstance(m, AFSC_DCT))
                total_loss = loss + mask_reg_weight * mask_penalty

            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += total_loss.item() * imgs.size(0)

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # ------------------------------
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                preds = model(imgs)
                val_preds.append(preds)
                val_targets.append(labels)

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_mse = mean_squared_error(val_preds, val_targets)
        val_rmse = torch.sqrt(val_mse).item()

        writer.add_scalar('Loss/val_rmse', val_rmse, epoch)

        log_masks_to_tensorboard(model, writer, epoch)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_afsc_dct_model.pth")
            print(f"Saved new best model (RMSE={val_rmse:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"[Early Stop] Patience {patience} exceeded.")
                break

    # ------------------------------
    # Final Test Evaluation
    model.load_state_dict(torch.load("best_afsc_dct_model.pth"))
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            preds = model(imgs)
            test_preds.append(preds)
            test_targets.append(labels)

    test_preds = torch.cat(test_preds)
    test_targets = torch.cat(test_targets)
    test_mse = mean_squared_error(test_preds, test_targets)
    test_rmse = torch.sqrt(test_mse).item()

    print(f"[Test] RMSE: {test_rmse:.4f}")
    writer.add_scalar('Loss/test_rmse', test_rmse, epoch)
    log_masks_to_tensorboard(model, writer, epoch + 1, max_channels=12, prefix='AFSC_final')
    writer.close()

    return model


# Log test scalar and final masks to TensorBoard
    try:
        writer.add_scalar('Loss/test', test_loss, epoch)
    except NameError:
        writer.add_scalar('Loss/test', test_loss, 0)

    # Log final masks (larger grid)
    try:
        log_masks_to_tensorboard(model, writer, epoch + 1, max_channels=12, prefix='AFSC_final')
    except NameError:
        log_masks_to_tensorboard(model, writer, 0, max_channels=12, prefix='AFSC_final')

    # Close the writer
    writer.close()

    # Save a comprehensive checkpoint including mask_history so Manim / other tools can read it
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'low_freq_size': low_freq_size,
        'replace_layers': replace_layers,
        'mask_history': mask_history,
    }, 'afsc_dct_checkpoint.pth')

    return model, (train_losses, val_losses)

# ------------------------------
# If script is executed as main, run example training
# ------------------------------
if __name__ == "__main__":
    # Example usage settings (tune as needed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, losses = train_main(
        train_csv='train.csv',
        val_csv='val.csv',
        test_csv='test.csv',
        replace_layers=('conv1', 'layer1'),   # gradually replace conv1 and layer1 blocks
        low_freq_size=(24, 24),               # low-frequency block (24x24) - adjust smaller/larger
        mask_mode='per_out',
        input_size=(224, 224),
        batch_size=1,  # Reduced for debugging
        num_epochs=30,
        device=device,
        logdir='runs/afsc_dct'
    )