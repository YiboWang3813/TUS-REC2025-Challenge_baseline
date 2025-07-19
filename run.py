import os
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# Dataset and transforms
from tools.dataloaders import FullSweepFreehandUSDataset
from tools.data_transforms import ComposeTransform, NormalizeTransform, DownsampleTransform

# Network and training tools
from tools.networks import build_network
from tools.losses import PCCLoss, PointDistance
from tools.transforms import get_reference_image_points
from tools.mics import read_calibration_matrices, get_batch_size_for_n, get_n_for_epoch
from tools.tb_writers import TensorboardWriter
from engines import train_one_epoch, validate_one_epoch


def train(args):
    device = torch.device('cuda:0')

    # Build model, loss functions, optimizer
    network = build_network(args).to(device)
    criterion_loss_mse = torch.nn.MSELoss()
    criterion_loss_pcc = PCCLoss()
    criterion_metric = PointDistance()
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    # Reference points in image space
    image_shape = [int(n) for n in args.image_shape.split(',')]
    ref_points = get_reference_image_points(image_shape, density=2).to(device)

    print(f"[INFO] Network: {args.network_name} initialized.")

    # Compose transforms for training/validation
    common_transforms = ComposeTransform([
        DownsampleTransform(2),
        NormalizeTransform(),
    ])

    # Use initial n_choices for dataset building
    init_n = get_n_for_epoch(0)

    # Build datasets with full subvolume sweep and fold split
    dataset_train = FullSweepFreehandUSDataset(
        src_dir=args.dataset_dir,
        n=init_n,
        fold=args.fold,
        mode='train',
        transforms=common_transforms
    )
    dataset_valid = FullSweepFreehandUSDataset(
        src_dir=args.dataset_dir,
        n=init_n,
        fold=args.fold,
        mode='validate',
        transforms=common_transforms
    )

    # Load calibration matrices
    calib_path = os.path.join(args.dataset_dir, 'calib_matrix.csv')
    tform_image_pixel_to_mm, tform_image_mm_to_tool = read_calibration_matrices(calib_path)
    tform_image_pixel_to_mm = tform_image_pixel_to_mm.to(device)
    tform_image_mm_to_tool = tform_image_mm_to_tool.to(device)

    # Setup experiment directory per fold
    exp_name = f'{args.network_name}_fold{args.fold}'
    args.exp_dir = os.path.join(args.checkpoints_dir, exp_name)
    os.makedirs(args.exp_dir, exist_ok=True)

    writer = TensorboardWriter(args.exp_dir)

    best_dist_valid = float('inf')
    # best_loss_valid = float('inf')

    for epoch in range(args.n_epochs + 1):
        # Set n_choices and batch_size for current epoch
        # current_n = get_n_for_epoch(epoch)
        current_n = 4 
        dataset_train.update_n_and_rebuild(current_n)
        dataset_valid.update_n_and_rebuild(current_n)

        batch_size = get_batch_size_for_n(current_n)

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=batch_size,
            num_workers=8,
            shuffle=True
        )

        print(f"[TRAIN] Epoch {epoch} | n: {current_n} | batch_size: {batch_size}")

        # Train one epoch
        loss_train, dist_train = train_one_epoch(
            args, dataloader_train, network, optimizer,
            criterion_loss_mse, criterion_loss_pcc,
            criterion_metric, ref_points,
            tform_image_pixel_to_mm, tform_image_mm_to_tool, device
        )

        writer.print_info(epoch, loss_train, dist_train, 'train')

        # Run validation periodically
        if epoch % args.freq_val == 0 and epoch != 0:
            batch_size_val = get_batch_size_for_n(current_n)

            dataloader_val = DataLoader(
                dataset_valid,
                batch_size=batch_size_val,
                num_workers=8,
                shuffle=False
            )

            print(f"[VALID] Epoch {epoch} | n: {current_n} | batch_size: {batch_size_val}")

            loss_valid, dist_valid = validate_one_epoch(
                args, dataloader_val, network,
                criterion_loss_mse, criterion_loss_pcc,
                criterion_metric, ref_points,
                tform_image_pixel_to_mm, tform_image_mm_to_tool, device
            )

            writer.print_info(epoch, loss_valid, dist_valid, 'valid')

            # Save network checkpoint and metrics
            writer.save_network(network, args.network_name, f'epoch_{epoch}')
            writer.add_scalars({
                'loss_train': loss_train,
                'dist_train': dist_train,
                'loss_valid': loss_valid,
                'dist_valid': dist_valid,
            }, epoch)

            if dist_valid < best_dist_valid:
                best_dist_valid = dist_valid
                writer.save_network(network, args.network_name, 'best')

    # Save best metrics for summary
    metric_path = os.path.join(args.exp_dir, 'metrics_best.npz')
    np.savez(metric_path, dist_valid=best_dist_valid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/raid/liujie/code_recon/data/ultrasound/Freehand_US_data_train_2025', help='Path to dataset directory')
    parser.add_argument('--checkpoints_dir', type=str, default='/raid/liujie/code_recon/checkpoints', help='Directory to store checkpoints')
    parser.add_argument('--exp_dir', type=str, default='', help='Optional: override full experiment path')
    parser.add_argument('--image_shape', type=str, default='480,640', help='Image shape as H,W')
    parser.add_argument('--network_name', type=str, default='frame_pair_model')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epochs', type=int, default=200)
    # parser.add_argument('--freq_info', type=int, default=10)
    # parser.add_argument('--freq_save', type=int, default=10)
    parser.add_argument('--freq_val', type=int, default=20)
    parser.add_argument('--fold', type=int, default=0, help='Fold index (0~4)')

    args = parser.parse_args()
    train(args)
