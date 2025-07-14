import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.nn.utils.rnn import pad_sequence


class FramePairModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', pred_dim=6):
        super().__init__()

        # Load EfficientNet backbone
        if backbone_name == 'efficientnet_b0':
            # weights = EfficientNet_B0_Weights.DEFAULT
            model = efficientnet_b0(weights=None)  # 不加载权重
            model.load_state_dict(torch.load('/raid/liujie/code_recon/checkpoints/efficientnet_b0_rwightman-3dd342df.pth')) 
            # model.load_state_dict(torch.load("/raid/liujie/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-3dd342df.pth"), strict=False)
        else:
            raise ValueError("Only efficientnet_b0 is supported.")

        # Modify first conv to accept 2-channel input
        conv1 = model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=2,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias is not None
        )
        with torch.no_grad():
            new_conv.weight[:, :2] = conv1.weight[:, :2]
            if conv1.in_channels > 2:
                new_conv.weight[:, 2:] = 0  # zero-init unused channels

        model.features[0][0] = new_conv

        # Encoder and regressor
        self.encoder = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d(1)  # -> (B*(N-1), C, 1, 1)
        )
        self.feature_dim = model.classifier[1].in_features  # e.g., 1280
        self.regressor = nn.Sequential(
            nn.Flatten(),               # -> (B*(N-1), feature_dim)
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, pred_dim)   # -> (B*(N-1), 6)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, H, W) padded input volume
            lengths: (B,) actual valid frame count for each sample
        Returns:
            out: (B, max(N_i - 1), 6) with padding
        """
        B, N, H, W = x.shape
        preds = []

        for i in range(B):
            n = lengths[i].item()
            if n < 2:
                raise ValueError(f"Sample {i} has less than 2 valid frames.")

            # (n-1, H, W) pairs: (img_t, img_{t+1})
            img1 = x[i, :n - 1]
            img2 = x[i, 1:n]
            pair = torch.stack([img1, img2], dim=1)  # (n-1, 2, H, W)

            features = self.encoder(pair)            # (n-1, C, 1, 1)
            out = self.regressor(features)           # (n-1, 6)
            preds.append(out)

        # Pad all to same shape (B, max_n-1, 6)
        preds_padded = pad_sequence(preds, batch_first=True)  # (B, max_n-1, 6)
        return preds_padded


def build_network(args): 
    if args.network_name == 'frame_pair_model': 
        return FramePairModel() 
    else: 
        raise NotImplementedError(f'{args.network_name} is not implemented.')
