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


class PCCLoss(nn.Module):
    """
    Pearson Correlation Coefficient loss.

    Args:
        split_dof (bool): If True, compute PCC separately over each of the 6 DoF and average.
    """
    def __init__(self, split_dof: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_dof = split_dof

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Tensor of shape (B, N-1, 6)
            y_true: Tensor of shape (B, N-1, 6)
        Returns:
            loss: scalar loss = 1 - PCC
        """

        def pcc_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """
            Compute Pearson correlation coefficient between two tensors.
            """
            xy = x * y
            mean_x = x.mean()
            mean_y = y.mean()
            cov_xy = xy.mean() - mean_x * mean_y

            std_x = x.std()
            std_y = y.std()

            pcc = cov_xy / (std_x * std_y + 1e-8)  # avoid divide-by-zero
            return pcc

        if self.split_dof:
            # Compute PCC for each DoF dimension and average
            pcc_total = 0
            for i in range(y_true.shape[-1]):
                pcc_total += pcc_func(y_true[..., i], y_pred[..., i])
            pcc_mean = pcc_total / y_true.shape[-1]
        else:
            # Flatten all values and compute single PCC
            pcc_mean = pcc_func(y_true.flatten(), y_pred.flatten())

        return 1 - pcc_mean
