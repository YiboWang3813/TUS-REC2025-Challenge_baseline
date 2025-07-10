import os 
import torch 
import json 
import random 
import argparse 
from torch.utils.data import DataLoader


from tools.dataloaders import FreehandUSRecDataset2025, pad_collate_fn 
from tools.data_transforms import * 

from tools.networks import build_network 
from tools.losses import * 

from tools.transforms import 


def get_parsed_arguments(parser: argparse.ArgumentParser): 
    parser.add_argument('--dataset_dir', type=str, default='/raid/liujie/code_recon/data/ultrasound/Freehand_US_data_train_2025', help='path to store dataset')
    parser.add_argument('--checkpoints_dir', type='str', default='/raid/liujie/code_recon/checkpoints', help='path to store trained weights and metrics')
    parser.add_argument('--exp_dir', type=str, default='', help='experiment dir decided by exp name and checkpoints dir')
    
    parser.add_argument('--network_name', type='str', default='frame_pair_model', help='name to choose network') 

    parser.add_argument('--batch_size', type=int, default=1, help='initial batch size') 
    
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate') 
    parser.add_argument('--n_epochs', type=int, default=30, help='number of training epochs') 
    parser.add_argument('--freq_info', type=int, default=10, help='how many epochs to print info once') 
    parser.add_argument('--freq_save', type=int, default=10, help='how many epochs to save networks weights once')
    parser.add_argument('--freq_val', type=int, default=10, help='how many epochs to validate once') 

    args = parser.parse_args() 

    return args 


def get_exp_name(args): 
    exp_name = '' 
    exp_name += f'network_{args.network_name}-epochs_{args.n_epochs}'
    return exp_name 


def get_n_choices_for_epoch(epoch):
    if epoch < 5:
        return [2]
    elif epoch < 10:
        return [2, 4]
    elif epoch < 15:
        return [2, 4, 8]
    elif epoch < 20: 
        return [2, 4, 8, 16]
    elif epoch < 25: 
        return [2, 4, 8, 16, 32]


def get_batch_size_for_n(n, base_batch=8, base_n=16):
    return max(1, base_batch * base_n // n)


def train_one_epoch(args, dataloader, network, optimizer, criterion_loss, criterion_metric, device): 
    this_epoch_loss, this_epoch_dist = 0, 0 
    for step, (frames, tforms, _) in enumerate(dataloader):
        frames, tforms = frames.to(device), tforms.to(device)

        # transform label based on label type 把tforms转换为label 
        # tforms是一个batch的变换矩阵 每个batch里有num_samples个变换矩阵 (batch_size, num_samples, 4, 4) 
        # tforms的变换关系是 从tracker tool space to camera space 这里camera space只是一个中介
        # 通过他转换得到 在tracker tool space下 tool1到tool0的变换 再得到在image space下 image1到image0的变换
        # 最后通过image points 得到在这个batch中 所有pair里 4个参考点的位置 shape (batch_size, num_pairs, 3, num_image_points)
        labels = transform_label(tforms, tforms_inv)

        optimiser.zero_grad()
        # model prediction
        outputs = model(frames)
        # transform prediction according to label type
        preds = transform_prediction(outputs)
        # calculate loss
        loss = criterion(preds, labels)
        loss.backward()
        optimiser.step()

        # transfrom prediction and label into points, for metric calculation
        preds_pts = transform_into_points(preds.data)
        labels_pts = transform_into_points(labels)
        dist = metrics(preds_pts, labels_pts).detach()
    
        train_epoch_loss += loss.item()
        train_epoch_dist += dist
        
    train_epoch_loss /= (step + 1)
    train_epoch_dist /= (step + 1)


def train(args):
    device = torch.device('cuda:0')  
    # build network, criterion, optimizer 
    network = build_network(args) 
    network.to(device) 
    criterion_loss = torch.nn.MSELoss() 
    criterion_metric = PointDistance()
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

    for epoch in range(args.n_epochs): 
        # build specific dataloader for this epoch 
        dataset_train.n_choices = get_n_choices_for_epoch(epoch) 
        n_sample = random.choice(dataset_train.n_choices)
        batch_size = get_batch_size_for_n(n_sample)

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                                      shuffle=True, collate_fn=pad_collate_fn)

        print(f"Epoch {epoch} | n_choices: {dataset_train.n_choices} | batch_size: {batch_size}")

        train_one_epoch(args, dataloader_train, network, optimizer,
                        criterion_loss, criterion_metric, device) 




     

if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    args = get_parsed_arguments(parser) 
    exp_name = get_exp_name(args) 
    args.exp_dir = os.path.join(args.checkpoints_dir, exp_name) 