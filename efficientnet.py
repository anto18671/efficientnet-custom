import torch.nn.functional as F
import torch.nn as nn
import torch


# ---------------------------
# Squeeze-and-Excitation block
# ---------------------------
class SqueezeExcite(nn.Module):
    # Initialize SE block
    def __init__(self, input_channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, input_channels // reduction)

        # First squeeze layer
        self.fc1 = nn.Conv2d(input_channels, reduced_channels, 1)

        # Second excite layer
        self.fc2 = nn.Conv2d(reduced_channels, input_channels, 1)

    # Forward pass
    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


# ---------------------------
# MBConv block
# ---------------------------
class MBConv(nn.Module):
    # Initialize MBConv
    def __init__(self, in_channels, out_channels, stride, expand_ratio, reduction=4, bn_eps=1e-3, bn_mom=0.01, kernel_size=3, dropout=0.0):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        use_expansion = expand_ratio != 1
        self.use_residual = (stride == 1 and in_channels == out_channels)

        # Layers container
        layers = []

        # Expansion pointwise convolution
        if use_expansion:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim, eps=bn_eps, momentum=bn_mom))
            layers.append(nn.SiLU(inplace=True))

        # Depthwise convolution
        dw_in = hidden_dim if use_expansion else in_channels
        layers.append(nn.Conv2d(dw_in, dw_in, kernel_size, stride, kernel_size // 2, groups=dw_in, bias=False))
        layers.append(nn.BatchNorm2d(dw_in, eps=bn_eps, momentum=bn_mom))
        layers.append(nn.SiLU(inplace=True))

        # Squeeze-and-Excitation
        layers.append(SqueezeExcite(dw_in, reduction=reduction))

        # Projection pointwise convolution
        layers.append(nn.Conv2d(dw_in, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_mom))

        # Sequential block
        self.block = nn.Sequential(*layers)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Find last BatchNorm for residual scaling
        proj_bn = None
        for m in reversed(self.block):
            if isinstance(m, nn.BatchNorm2d):
                proj_bn = m
                break

        # Zero-initialize last BN gamma when residual is used
        if self.use_residual and proj_bn is not None:
            nn.init.zeros_(proj_bn.weight)

    # Forward pass
    def forward(self, x):
        out = self.block(x)
        out = self.dropout(out)
        if self.use_residual:
            return out + x
        return out


# ---------------------------
# EfficientNet backbone
# ---------------------------
class EfficientNet(nn.Module):
    # Initialize EfficientNet
    def __init__(self, base_cfg, num_classes=1000, width_mult=1.0, depth_mult=1.0, bn_eps=1e-3, bn_mom=0.01, dropout=0.2, se_reduction=4, kernel_size=3):
        super().__init__()

        # Base configuration stored
        self.base_cfg = base_cfg

        # Rounded filters helper
        def round_filters(c):
            scaled = c * width_mult
            new_c = max(8, int(scaled + 4) // 8 * 8)
            if new_c < 0.9 * scaled:
                new_c += 8
            return int(new_c)

        # Rounded repeats helper
        def round_repeats(n):
            return int(torch.ceil(torch.tensor(n * depth_mult)).item())

        # Stem out channels
        stem_out = round_filters(32)

        # Stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_out, eps=bn_eps, momentum=bn_mom),
            nn.SiLU(inplace=True),
        )

        # Blocks container
        blocks = []
        in_ch = stem_out

        # Build blocks from base configuration
        for t, c, n, s in self.base_cfg:
            out_ch = round_filters(c)
            reps = round_repeats(n)
            for i in range(reps):
                stride = s if i == 0 else 1
                blocks.append(
                    MBConv(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        stride=stride,
                        expand_ratio=t,
                        reduction=se_reduction,
                        bn_eps=bn_eps,
                        bn_mom=bn_mom,
                        kernel_size=kernel_size,
                        dropout=0.0
                    )
                )
                in_ch = out_ch

        # Stacked blocks
        self.blocks = nn.Sequential(*blocks)

        # Head out channels
        head_out = round_filters(1024)

        # Head projection
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_out, 1, bias=False),
            nn.BatchNorm2d(head_out, eps=bn_eps, momentum=bn_mom),
            nn.SiLU(inplace=True),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(head_out, num_classes, bias=True),
        )

        # Initialize weights
        self.init_weights()

    # Initialize all module weights
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # Forward pass
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.classifier(x)
        return x
