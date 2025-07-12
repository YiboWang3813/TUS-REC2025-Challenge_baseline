import os 
import torch 

from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter: 
    def __init__(self, exp_dir: str): 
        self.writer = SummaryWriter(exp_dir) 

        # Build necessary sub dirs 
        self.exp_dir = exp_dir 
        self.weights_dir = os.path.join(exp_dir, 'weights') 
        os.makedirs(self.weights_dir, exist_ok=True) 
    
    def print_info(self, epoch, loss, dist, mode='train'): 
        msg = f'[Epoch {epoch}] {mode}-loss={loss:.3f}, {mode}-dist={dist:.3f}' 
        print(msg) 

    def save_network(self, network, network_name, suffix=''): 
        save_name = '{}_{}.pth'.format(network_name, suffix) 
        torch.save(network.state_dict(), os.path.join(self.weights_dir, save_name)) 
        print(f'{network_name} save {suffix} done') 

    def add_scalars(self, scalars_dict, epoch): 
        for k, v in scalars_dict.items(): 
            if k.find('loss') != -1: 
                self.writer.add_scalar('loss', {k: v}, epoch) 
            elif k.find('dist') != -1: 
                self.writer.add_scalar('dist', {k: v}, epoch)  