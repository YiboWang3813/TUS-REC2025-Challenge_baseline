import os
import torch
import h5py
from torch.utils.data import Dataset


class FullSweepFreehandUSDataset(Dataset):
    def __init__(self, src_dir, n, fold=0, mode='train', transforms=None):
        """
        Args:
            src_dir (str): dataset root dir
            n (int): fixed number of frames per sample
            fold (int): 0~4 for 5-fold cross-validation
            mode (str): 'train' or 'validate'
        """
        assert mode in ['train', 'validate']
        self.src_dir = src_dir
        self.transforms = transforms
        self.fold = fold
        self.mode = mode
        self.n = n
        self.subvolume_specs = []
        self.data_entries = []

        self._load_data_paths()
        self._rebuild_subvolume_indices()

    def _load_data_paths(self):
        work_dir = os.path.join(self.src_dir, 'frames_transfs')
        subject_names = sorted(os.listdir(work_dir))
        object_names = ['LH_rotation.h5', 'RH_rotation.h5']

        # 5-fold split
        fold_size = len(subject_names) // 5
        val_start = self.fold * fold_size
        val_end = val_start + fold_size

        if self.mode == 'train':
            selected = subject_names[:val_start] + subject_names[val_end:]
        else:
            selected = subject_names[val_start:val_end]

        for subject in selected:
            for obj in object_names:
                path = os.path.join(work_dir, subject, obj)
                if os.path.exists(path):
                    self.data_entries.append(path)
                else:
                    print(f"[WARNING] Missing file: {path}")

    def _rebuild_subvolume_indices(self):
        self.subvolume_specs.clear()
        for sample_idx, path in enumerate(self.data_entries):
            with h5py.File(path, 'r') as f:
                N = f['frames'].shape[0]

            if self.n > N:
                continue

            for start_idx in range(N - self.n + 1):
                self.subvolume_specs.append((sample_idx, start_idx))

    def update_n_and_rebuild(self, new_n: int):
        """Update number of frames per sample and rebuild slices"""
        self.n = new_n
        self._rebuild_subvolume_indices()

    def __len__(self):
        return len(self.subvolume_specs)

    def __getitem__(self, index):
        sample_idx, start_idx = self.subvolume_specs[index]
        path = self.data_entries[sample_idx]

        with h5py.File(path, 'r') as f:
            frames = torch.tensor(f['frames'][start_idx:start_idx + self.n])     # (n, H, W)
            tforms = torch.tensor(f['tforms'][start_idx:start_idx + self.n])     # (n, 4, 4)

        if self.transforms is not None:
            frames = self.transforms(frames)

        return frames, tforms, self.n
