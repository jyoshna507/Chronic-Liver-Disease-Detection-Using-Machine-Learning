import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class PrimaryCapsule(nn.Module):
    def __init__(self, in_channels, out_channels, capsule_dim, kernel_size=3, stride=1, padding=1):
        super(PrimaryCapsule, self).__init__()
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(in_channels, out_channels * capsule_dim, kernel_size, stride, padding)

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)

    def forward(self, x):
        x = self.conv(x)
        B, C_dim, H, W = x.size()
        x = x.view(B, -1, self.capsule_dim, H, W)
        return self.squash(x, dim=2)

class CapsuleRouting(nn.Module):
    def __init__(self, in_capsules, out_capsules, in_dim, out_dim):
        super(CapsuleRouting, self).__init__()
        # Simplified robust routing using an affine transform
        self.conv = nn.Conv2d(in_capsules * in_dim, out_capsules * out_dim, 1)
        self.out_capsules = out_capsules
        self.out_dim = out_dim

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)

    def forward(self, x):
        B, _, _, H, W = x.size()
        x = x.view(B, -1, H, W)
        x = self.conv(x)
        x = x.view(B, self.out_capsules, self.out_dim, H, W)
        return self.squash(x, dim=2)

class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(DilatedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class Model1(nn.Module):
    def __init__(self, num_classes=3):
        super(Model1, self).__init__()
        
        # Encoder: ResNet-50
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Change first conv to accept 1 channel (CT)
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1 # 256 output
        self.layer2 = resnet.layer2 # 512 output
        self.layer3 = resnet.layer3 # 1024 output
        self.layer4 = resnet.layer4 # 2048 output

        # Capsule Representation
        self.primary_caps = PrimaryCapsule(2048, 32, 16)
        self.routing = CapsuleRouting(in_capsules=32, out_capsules=16, in_dim=16, out_dim=32)

        # Decoder
        self.dec4 = DilatedResidualBlock(16*32, 512, dilation=2)
        self.dec3 = DilatedResidualBlock(512 + 1024, 256, dilation=4)
        self.dec2 = DilatedResidualBlock(256 + 512, 128, dilation=2)
        self.dec1 = DilatedResidualBlock(128 + 256, 64, dilation=1)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        u = self.primary_caps(x4)
        v = self.routing(u)
        B, C, D, H, W = v.size()
        v = v.view(B, C*D, H, W)

        # Classifier logits (from capsule layer pooled)
        d4 = self.dec4(v)
        cls_feat = self.gap(d4).view(B, -1)
        logits = self.classifier(cls_feat)

        d4 = F.interpolate(d4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        d3 = F.interpolate(self.dec3(torch.cat([d4, x3], dim=1)), size=x2.shape[2:], mode='bilinear', align_corners=True)
        d2 = F.interpolate(self.dec2(torch.cat([d3, x2], dim=1)), size=x1.shape[2:], mode='bilinear', align_corners=True)
        d1 = F.interpolate(self.dec1(torch.cat([d2, x1], dim=1)), size=x.shape[2:], mode='bilinear', align_corners=True)

        mask = self.final_conv(d1)
        # ensure matching input size
        mask = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=True)

        return mask, logits
