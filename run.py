import os
import random
import argparse
import torch
from torch.utils.data import DataLoader

# Dataset and transforms
from tools.dataloaders import FreehandUSRecDataset2025, pad_collate_fn
from tools.data_transforms import ComposeTransform, NormalizeTransform, DownsampleTransform

# Network and training tools
from tools.networks import build_network
from tools.losses import MaskedMSELoss, MaskedPCCLoss, PointDistance
from tools.transforms import get_reference_image_points
from tools.mics import read_calibration_matrices, get_batch_size_for_n, get_n_choices_for_epoch
from tools.tb_writers import TensorboardWriter
from engines import train_one_epoch, validate_one_epoch


def train(args):
    device = torch.device('cuda:0')

    # Build model, loss functions, optimizer
    network = build_network(args).to(device)
    criterion_loss_mse = MaskedMSELoss()
    criterion_loss_pcc = MaskedPCCLoss()
    criterion_metric = PointDistance()
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    # Reference points in image space
    image_shape = [int(n) for n in args.image_shape.split(',')]
    ref_points = get_reference_image_points(image_shape, density=2)

    print(f"[INFO] Network: {args.network_name} initialized.")

    # Compose transforms for training/validation
    common_transforms = ComposeTransform([
        DownsampleTransform(2),
        NormalizeTransform(),
    ])

    # Build datasets
    dataset_train = FreehandUSRecDataset2025(args.dataset_dir, [2], mode='train', transform=common_transforms)
    dataset_valid = FreehandUSRecDataset2025(args.dataset_dir, [2], mode='validate', transform=common_transforms)

    # Load calibration matrices
    calib_path = os.path.join(args.dataset_dir, 'calib_mat.csv')  # TODO: Confirm this path is correct
    tform_image_pixel_to_mm, tform_image_mm_to_tool = read_calibration_matrices(calib_path)
    tform_image_pixel_to_mm = tform_image_pixel_to_mm.to(device)
    tform_image_mm_to_tool = tform_image_mm_to_tool.to(device)

    # TensorBoard writer
    writer = TensorboardWriter(args.exp_dir)

    best_metric_valid = float('inf')

    for epoch in range(args.n_epochs):
        # Dynamically set n_choices and batch_size for training
        dataset_train.n_choices = get_n_choices_for_epoch(epoch)
        n_sample = random.choice(dataset_train.n_choices)
        batch_size = get_batch_size_for_n(n_sample)

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                                      shuffle=True, collate_fn=pad_collate_fn)

        print(f"[TRAIN] Epoch {epoch} | n_choices: {dataset_train.n_choices} | batch_size: {batch_size}")

        # Train one epoch
        loss_train, metric_train = train_one_epoch(
            args, dataloader_train, network, optimizer,
            criterion_loss_mse, criterion_loss_pcc,
            criterion_metric, ref_points,
            tform_image_pixel_to_mm, tform_image_mm_to_tool, device
        )

        writer.print_info(epoch, loss_train, metric_train, 'train')

        # Run validation periodically
        if epoch % args.freq_val == 0:
            dataset_valid.n_choices = get_n_choices_for_epoch(epoch)
            n_sample_val = random.choice(dataset_valid.n_choices)
            batch_size_val = get_batch_size_for_n(n_sample_val)

            dataloader_val = DataLoader(dataset_valid, batch_size=batch_size_val,
                                        shuffle=False, collate_fn=pad_collate_fn)

            print(f"[VALID] Epoch {epoch} | n_choices: {dataset_valid.n_choices} | batch_size: {batch_size_val}")

            loss_valid, metric_valid = validate_one_epoch(
                args, dataloader_val, network,
                criterion_loss_mse, criterion_loss_pcc,
                criterion_metric, ref_points,
                tform_image_pixel_to_mm, tform_image_mm_to_tool, device
            )

            writer.print_info(epoch, loss_valid, metric_valid, 'valid')

            # Save network checkpoint and metrics
            writer.save_network(network, args.network_name, f'epoch_{epoch}')
            writer.add_scalars({
                'loss_train': loss_train.item(),
                'dist_train': metric_train.item(),
                'loss_valid': loss_valid.item(),
                'dist_valid': metric_valid.item(),
            }, epoch)

            # Save best performing model
            if metric_valid < best_metric_valid:
                best_metric_valid = metric_valid
                writer.save_network(network, args.network_name, 'best')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/raid/liujie/code_recon/data/ultrasound/Freehand_US_data_train_2025',
                        help='Path to dataset directory')
    parser.add_argument('--checkpoints_dir', type=str, default='/raid/liujie/code_recon/checkpoints',
                        help='Directory to store model checkpoints')
    parser.add_argument('--exp_dir', type=str, default='',
                        help='Full path to experiment directory')
    parser.add_argument('--image_shape', type=str, default='480,640',
                        help='Image shape as H,W (e.g. 480,640)')
    parser.add_argument('--network_name', type=str, default='frame_pair_model',
                        help='Name of the model architecture')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Initial batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--freq_info', type=int, default=10,
                        help='Print training info every n epochs')
    parser.add_argument('--freq_save', type=int, default=10,
                        help='Save model checkpoint every n epochs')
    parser.add_argument('--freq_val', type=int, default=10,
                        help='Run validation every n epochs')

    args = parser.parse_args()

    # Auto-generate experiment directory name if not set
    exp_name = f'network_{args.network_name}-epochs_{args.n_epochs}'
    args.exp_dir = os.path.join(args.checkpoints_dir, exp_name)

    train(args)
