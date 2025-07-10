import os 

from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter: 
    def __init__(self, exp_dir: str): 
        self.writer = SummaryWriter(exp_dir) 
    
    