import os
import torch
import h5py
import random
from torch.utils.data import Dataset

class FreehandUSRecDataset2025(Dataset):
    def __init__(self, src_dir, n_choices, mode='train', transforms=None):
        """
        Args:
            src_dir (str): root directory containing 'frames_transfs'
            n_choices (List[int]): possible window sizes for sampling
            mode (str): 'train' or 'validate'
        """
        assert mode in ['train', 'validate']
        self.n_choices = n_choices
        self.object_paths = []

        work_dir = os.path.join(src_dir, 'frames_transfs')
        subject_names = sorted(os.listdir(work_dir))
        object_names = ['LH_rotation.h5', 'RH_rotation.h5']

        if mode == 'train':
            selected = subject_names[:-5]
        else:
            selected = subject_names[-5:]

        for subject in selected:
            for obj in object_names:
                path = os.path.join(work_dir, subject, obj)
                if os.path.exists(path):
                    self.object_paths.append(path)
                else:
                    print(f"Warning: {path} not found.")
        
        self.transforms = transforms

    def __len__(self):
        return len(self.object_paths)

    def __getitem__(self, index):
        path = self.object_paths[index]
        with h5py.File(path, 'r') as f:
            frames = torch.tensor(f['frames'][:])     # (N, H, W)
            tforms = torch.tensor(f['tforms'][:])     # (N, 4, 4)

        N = frames.shape[0]
        n = random.choice(self.n_choices)
        if n > N:
            raise ValueError(f"n = {n} > total frames = {N}")

        start = random.randint(0, N - n)
        subvolume = frames[start:start + n]           # (n, H, W)
        subtforms = tforms[start:start + n]           # (n, 4, 4)

        if self.transforms is not None: 
            subvolume = self.transforms(subvolume)

        return subvolume, subtforms, n

def pad_collate_fn(batch):
    """
    Args:
        batch: list of (frames: [n, H, W], tforms: [n, 4, 4], n)
    Returns:
        padded_frames: [B, n_max, H, W]
        padded_tforms: [B, n_max, 4, 4]
        lengths: [B] actual n values
    """
    frames_list = [item[0] for item in batch]
    tforms_list = [item[1] for item in batch]
    lengths = [item[2] for item in batch]

    # Pad along n axis (dim=0)
    padded_frames = torch.nn.utils.rnn.pad_sequence(frames_list, batch_first=True)    # [B, n_max, H, W]
    padded_tforms = torch.nn.utils.rnn.pad_sequence(tforms_list, batch_first=True)    # [B, n_max, 4, 4]

    return padded_frames, padded_tforms, torch.tensor(lengths)
