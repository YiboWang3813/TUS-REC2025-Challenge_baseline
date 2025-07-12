import os 
import torch 
import json 
import random 
import argparse 
from torch.utils.data import DataLoader

from tools.dataloaders import FreehandUSRecDataset2025, pad_collate_fn 
from tools.data_transforms import ComposeTransform, NormalizeTransform, DownsampleTransform 
from tools.networks import build_network 
from tools.losses import MaskedMSELoss, MaskedPCCLoss, PointDistance 
from tools.transforms import get_reference_image_points
from tools.losses import MaskedMSELoss, MaskedPCCLoss 
from tools.mics import read_calibration_matrices, get_batch_size_for_n, get_n_choices_for_epoch 
from tools.tb_writers import TensorboardWriter 

from engines import train_one_epoch, validate_one_epoch 


def train(args):
    device = torch.device('cuda:0')  
    # build network, criterion, optimizer 
    network = build_network(args) 
    network.to(device) 

    criterion_loss_mse = MaskedMSELoss() 
    criterion_loss_pcc = MaskedPCCLoss() 

    criterion_metric = PointDistance()
    ref_points = get_reference_image_points([int(n) for n in args.image_shape.split(',')], 2) 

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr) 
    print(f'build network, name: {args.network_name}, criterion and optimizer done') 

    # build dataset 
    transforms_train = ComposeTransform([
        DownsampleTransform(2),
        NormalizeTransform(),
    ])
    dataset_train = FreehandUSRecDataset2025(args.dataset_dir, [2], 'train', transforms_train)
    transforms_valid = ComposeTransform([
        DownsampleTransform(2),
        NormalizeTransform(),
    ])
    dataset_valid = FreehandUSRecDataset2025(args.dataset_dir, [2], 'validate', transforms_valid) 

    # load necessary global tranforms 
    calib_path = os.path.join(args.src_dir, 'calib_mat.csv') # TODO 检查这个路径  
    tform_image_pixel_to_mm, tform_image_mm_to_tool = read_calibration_matrices(calib_path) 
    tform_image_pixel_to_mm = tform_image_pixel_to_mm.to(device) 
    tform_image_mm_to_tool = tform_image_mm_to_tool.to(device) 

    # initialize tensorboard writer 
    writer = TensorboardWriter(args.exp_dir)  

    best_metric_valid = 1e10 
    for epoch in range(args.n_epochs): 
        # build specific dataloader for this epoch 
        dataset_train.n_choices = get_n_choices_for_epoch(epoch) 
        n_sample = random.choice(dataset_train.n_choices)
        batch_size = get_batch_size_for_n(n_sample)

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                                      shuffle=True, collate_fn=pad_collate_fn)

        print(f"Epoch {epoch} | n_choices: {dataset_train.n_choices} | batch_size: {batch_size}")

        loss_train, metric_train = train_one_epoch(args, dataloader_train, network, optimizer,
                                                    criterion_loss_mse, criterion_loss_pcc, 
                                                    criterion_metric, ref_points, 
                                                    tform_image_pixel_to_mm, tform_image_mm_to_tool, 
                                                    device) 
        
        writer.print_info(epoch, loss_train, metric_train, 'train') 

        if epoch % args.freq_val == 0: 
            # Build dataloader for validation
            dataset_valid.n_choices = get_n_choices_for_epoch(epoch)  # 如果你希望验证集也支持不同长度，可以保留这一句
            n_sample_val = random.choice(dataset_valid.n_choices)      # 随机选择一个片段长度
            batch_size_val = get_batch_size_for_n(n_sample_val)      # 根据长度设置验证 batch size

            dataloader_val = DataLoader(
                dataset_valid,
                batch_size=batch_size_val,
                shuffle=False,  # 验证集不打乱顺序
                collate_fn=pad_collate_fn
            )

            print(f"[VALID] Epoch {epoch} | n_choices: {dataset_valid.n_choices} | batch_size: {batch_size_val}")

            loss_valid, metric_valid = validate_one_epoch(args, dataloader_val, network, 
                                                          criterion_loss_mse, criterion_loss_pcc, 
                                                          criterion_metric, ref_points, 
                                                          tform_image_pixel_to_mm, tform_image_mm_to_tool, device) 

            writer.print_info(epoch, loss_valid, metric_valid, 'valid') 

            # Save model and add scalars per validation epoch 
            writer.save_network(network, args.network_name, f'epoch_{epoch}') 
            writer.add_scalars({
                'loss_train': loss_train.item(), 'dist_train': metric_train.item(), 
                'loss_valid': loss_valid.item(), 'dist_valid': metric_valid.item(), 
            }, epoch)  

            # Re-compute best validation metric and save the best network 
            if metric_valid < best_metric_valid: 
                best_metric_valid = metric_valid 
                writer.save_network(network, args.network_name, 'best') 
            

if __name__ == '__main__': 
    # set arguments 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset_dir', type=str, default='/raid/liujie/code_recon/data/ultrasound/Freehand_US_data_train_2025', help='path to store dataset')
    parser.add_argument('--checkpoints_dir', type='str', default='/raid/liujie/code_recon/checkpoints', help='path to store trained weights and metrics')
    parser.add_argument('--exp_dir', type=str, default='', help='experiment dir decided by exp name and checkpoints dir')
    parser.add_argument('--image_shape', type=str, default='480,640', help='shape of image, (height, width)') 
    
    parser.add_argument('--network_name', type='str', default='frame_pair_model', help='name to choose network') 

    parser.add_argument('--batch_size', type=int, default=1, help='initial batch size') 
    
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate') 
    parser.add_argument('--n_epochs', type=int, default=30, help='number of training epochs') 
    parser.add_argument('--freq_info', type=int, default=10, help='how many epochs to print info once') 
    parser.add_argument('--freq_save', type=int, default=10, help='how many epochs to save networks weights once')
    parser.add_argument('--freq_val', type=int, default=10, help='how many epochs to validate once') 

    args = parser.parse_args()  

    # set exp name 
    exp_name = '' 
    exp_name += f'network_{args.network_name}-epochs_{args.n_epochs}'

    args.exp_dir = os.path.join(args.checkpoints_dir, exp_name) 

    train(args) 