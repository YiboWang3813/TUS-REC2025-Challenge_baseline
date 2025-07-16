import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
# from torchvision.models import EfficientNet_B0_Weights  # 已禁用权重下载


class FramePairModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', pred_dim=6):
        super().__init__()

        # Load EfficientNet backbone
        if backbone_name == 'efficientnet_b0':
            model = efficientnet_b0(weights=None)
            model.load_state_dict(torch.load('/raid/liujie/code_recon/checkpoints/efficientnet_b0_rwightman-3dd342df.pth'))
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
            nn.AdaptiveAvgPool2d(1)
        )
        self.feature_dim = model.classifier[1].in_features
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, pred_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, H, W) input volume
        Returns:
            out: (B, N-1, 6)
        """
        B, N, H, W = x.shape
        if N < 2:
            raise ValueError(f"Input sequence length N={N} must be at least 2.")

        # Generate image pairs: (img_t, img_{t+1}) along N axis
        img1 = x[:, :-1]  # (B, N-1, H, W)
        img2 = x[:, 1:]   # (B, N-1, H, W)
        pair = torch.stack([img1, img2], dim=2)  # (B, N-1, 2, H, W)
        pair = pair.view(-1, 2, H, W)            # (B*(N-1), 2, H, W)

        # Forward pass
        features = self.encoder(pair)            # (B*(N-1), C, 1, 1)
        out = self.regressor(features)           # (B*(N-1), 6)
        out = out.view(B, N - 1, -1)             # (B, N-1, 6)

        return out


def build_network(args):
    if args.network_name == 'frame_pair_model':
        return FramePairModel()
    else:
        raise NotImplementedError(f'{args.network_name} is not implemented.')
