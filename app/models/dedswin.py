import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=2)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        x = self.encoder(x)
        x = x.permute(1, 2, 0).view(b, c, h, w)
        return x

class FusionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)  # 128 + 128 = 256
        return self.conv(x)

class Model2(nn.Module):
    def __init__(self, num_classes=3):
        super(Model2, self).__init__()
        
        # CNN Encoder
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Transformer Branch
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, 128, 4, stride=4),
            nn.ReLU()
        )
        self.transformer = TransformerEncoder(dim=128)
        
        self.fusion = FusionDecoder()
        self.final_conv = nn.Conv2d(64, num_classes, 1)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn_encoder(x) # [B, 128, H/2, W/2]
        
        trans_feat = self.patch_embed(x) # [B, 128, H/4, W/4]
        trans_feat = self.transformer(trans_feat)
        
        # Upsample transformer feature map to match CNN
        trans_feat_up = F.interpolate(trans_feat, size=cnn_feat.shape[2:], mode='bilinear', align_corners=True)
        
        # Fusion
        fused = self.fusion(cnn_feat, trans_feat_up)
        mask = self.final_conv(fused)
        mask = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Classification logit
        c_feat = self.gap(cnn_feat).view(x.size(0), -1)
        t_feat = self.gap(trans_feat).view(x.size(0), -1)
        logits = self.classifier(torch.cat([c_feat, t_feat], dim=1))
        
        return mask, logits
