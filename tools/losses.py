import torch
import torch.nn as nn
import torch.nn.functional as F


class PointDistance:
    """
    Compute point-wise Euclidean distance loss between prediction and ground truth.
    Used when the prediction/label is a set of 3D points.

    Args:
        paired (bool): If True, computes distance pairwise; otherwise computes all-to-all.
    """
    def __init__(self, paired=True):
        self.paired = paired

    def __call__(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Tensor of shape (B, N-1, 3)
            labels: Tensor of shape (B, N-1, 3)
        Returns:
            loss: averaged Euclidean distance
        """
        dists = torch.sqrt(((preds - labels) ** 2).sum(dim=2))  # (B, N-1)
        return dists.mean()


class PCCLoss(nn.Module):
    """
    Pearson Correlation Coefficient loss on 4x4 transformation matrices (no masking).
    """
    def __init__(self, split_elements: bool = False):
        """
        Args:
            split_elements (bool): compute PCC per-matrix-entry and average
        """
        super().__init__()
        self.split_elements = split_elements

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: (B, N-1, 4, 4)
            y_true: (B, N-1, 4, 4)
        Returns:
            loss: scalar
        """
        # Flatten to (num_samples, 16)
        y_pred_flat = y_pred.view(-1, 16)
        y_true_flat = y_true.view(-1, 16)

        def pcc_func(x, y):
            xy = x * y
            mean_x = x.mean()
            mean_y = y.mean()
            cov_xy = xy.mean() - mean_x * mean_y
            std_x = x.std()
            std_y = y.std()
            return cov_xy / (std_x * std_y + 1e-8)

        if self.split_elements:
            pcc_total = 0
            for i in range(y_true_flat.shape[-1]):
                pcc_total += pcc_func(y_true_flat[:, i], y_pred_flat[:, i])
            pcc_mean = pcc_total / y_true_flat.shape[-1]
        else:
            pcc_mean = pcc_func(y_true_flat.flatten(), y_pred_flat.flatten())

        return 1 - pcc_mean

