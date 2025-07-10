import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class FramePairModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', pred_dim=6):
        super().__init__()

        # 加载 EfficientNet 预训练模型
        if backbone_name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.DEFAULT
            model = efficientnet_b0(weights=weights)
        else:
            raise ValueError("Only efficientnet_b0 is currently supported.")

        # 修改第一层输入通道为2，保留其余权重
        conv1 = model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=2,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias is not None
        )
        # 用 pretrained conv 权重初始化前两个通道
        with torch.no_grad():
            new_conv.weight[:, :2] = conv1.weight[:, :2]
            if conv1.in_channels > 2:
                # 多余通道初始化为0或均值（optional）
                new_conv.weight[:, 2:] = 0

        model.features[0][0] = new_conv

        # 去掉分类器，只保留特征提取部分
        self.encoder = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d(1)  # 输出 shape: (B*(N-1), C, 1, 1)
        )

        self.feature_dim = model.classifier[1].in_features  # e.g. 1280
        self.regressor = nn.Sequential(
            nn.Flatten(),               # (B*(N-1), feature_dim)
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, pred_dim)   # 输出变换参数，如 (B, N-1, 6)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, N, H, W)
        Returns:
            out: Tensor of shape (B, N-1, 6)
        """
        B, N, H, W = x.shape

        img1 = x[:, :-1]  # (B, N-1, H, W)
        img2 = x[:, 1:]   # (B, N-1, H, W)
        pairs = torch.stack([img1, img2], dim=2)  # (B, N-1, 2, H, W)
        pairs = pairs.view(B * (N - 1), 2, H, W)  # (B*(N-1), 2, H, W)

        features = self.encoder(pairs)           # (B*(N-1), C, 1, 1)
        out = self.regressor(features)           # (B*(N-1), 6)
        out = out.view(B, N - 1, -1)              # (B, N-1, 6)
        return out


def build_network(args): 
    net = None 
    if args.network_name == 'frame_pair_model': 
        net = FramePairModel() 
    else: 
        raise NotImplementedError(f'{args.network_name} has not been implemented') 
    return net 