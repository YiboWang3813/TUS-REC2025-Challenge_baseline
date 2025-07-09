import numpy as np
import torch
import torch.nn.functional as F

class DownsampleTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        # Convert numpy to torch if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.float()  # (N, H, W)
        x = x.unsqueeze(1)  # (N, 1, H, W)
        x = F.interpolate(
            x,
            scale_factor=1.0 / self.factor,
            mode='bilinear',
            align_corners=False
        )
        return x.squeeze(1)  # (N, h, w)


class NormalizeTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        # Convert numpy to torch if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.float()
        min_val = x.min()
        max_val = x.max()
        if max_val > min_val:
            x = (x - min_val) / (max_val - min_val)
        else:
            x = torch.zeros_like(x)  # avoid divide-by-zero
        return x


class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
