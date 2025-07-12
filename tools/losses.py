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
        # Paired Euclidean distance: sqrt(sum((x - y)^2)) over last dim
        dists = torch.sqrt(((preds - labels) ** 2).sum(dim=2))  # (B, N-1)
        
        if self.paired:
            # Mean over batch and time
            return dists.mean()
        else:
            # Optionally support non-paired cases (not used here)
            return dists.mean()


def mask_valid_entries_matrix(y_pred: torch.Tensor, y_true: torch.Tensor, lengths: torch.Tensor):
    """
    Extract valid entries (B, N-1, 4, 4) → (total_valid, 4, 4)
    Args:
        y_pred: (B, max_n-1, 4, 4)
        y_true: (B, max_n-1, 4, 4)
        lengths: (B,) number of valid frames
    Returns:
        y_pred_valid, y_true_valid: (total_valid, 4, 4)
    """
    B, max_n, _, _ = y_pred.shape
    mask = torch.zeros((B, max_n), dtype=torch.bool, device=y_pred.device)
    for i in range(B):
        n = lengths[i].item()
        if n > 1:
            mask[i, :n - 1] = True
    return y_pred[mask], y_true[mask]


class MaskedMSELoss(nn.Module):
    """
    MSE loss on 4x4 transformation matrices with masking.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: (B, N-1, 4, 4)
            y_true: (B, N-1, 4, 4)
            lengths: (B,)
        Returns:
            scalar loss
        """
        y_pred_valid, y_true_valid = mask_valid_entries_matrix(y_pred, y_true, lengths)
        return torch.mean((y_pred_valid - y_true_valid) ** 2)


class MaskedPCCLoss(nn.Module):
    """
    PCC loss on 4x4 transformation matrices with masking.
    """
    def __init__(self, split_elements: bool = False, *args, **kwargs):
        """
        Args:
            split_elements (bool): compute PCC per-matrix-entry and average
        """
        super().__init__(*args, **kwargs)
        self.split_elements = split_elements

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: (B, N-1, 4, 4)
            y_true: (B, N-1, 4, 4)
            lengths: (B,)
        Returns:
            loss: scalar
        """
        y_pred_valid, y_true_valid = mask_valid_entries_matrix(y_pred, y_true, lengths)
        # flatten to (num_valid, 16)
        y_pred_flat = y_pred_valid.view(y_pred_valid.shape[0], -1)
        y_true_flat = y_true_valid.view(y_true_valid.shape[0], -1)

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
