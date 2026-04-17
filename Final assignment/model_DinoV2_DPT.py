import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class DPTReassemble(nn.Module):
    """Projects and upsamples one ViT layer to a shared target resolution."""
    def __init__(self, embed_dim, out_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, out_dim, kernel_size=1)

    def forward(self, x, target_hw):
        x = self.proj(x)
        return F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)


class FusionHead(nn.Module):
    def __init__(self, channels=256, n_classes=19):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(channels, n_classes, kernel_size=1)

    def forward(self, features):
        fused = self.fusion(torch.cat(features, dim=1))
        return self.classifier(fused)


class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19, backbone='dinov2_vitb14'):
        super().__init__()
        self.n_classes = n_classes
        self.patch_size = 14

        base_dir = os.environ.get('DINOV2_BASE_DIR', '/app')
        os.environ['TORCH_HOME'] = base_dir

        self.backbone = torch.hub.load(
            os.path.join(base_dir, 'dinov2_hub'),
            backbone,
            source='local',
            pretrained=False,
        )

        state_dict = torch.load(
            os.path.join(base_dir, 'dinov2_vitb14_pretrain.pth'),
            map_location='cpu'
        )
        self.backbone.load_state_dict(state_dict, strict=False)

        # Frozen part
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last 6 blocks 
        for block in self.backbone.blocks[-6:]:
            for param in block.parameters():
                param.requires_grad = True

        # Unfreeze norm
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

        embed_dims = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
        }
        embed_dim = embed_dims[backbone]

        self.reassembles = nn.ModuleList([
            DPTReassemble(embed_dim, 256) for _ in range(4)
        ])

        self.head = FusionHead(channels=256, n_classes=n_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        # Target is 1/4 input resolution
        target_hw = (H // 4, W // 4)

        raw_features = self.backbone.get_intermediate_layers(x, n=[2, 5, 8, 11])

        projected = []
        for i, feat in enumerate(raw_features):
            f = feat.reshape(B, h_patches, w_patches, -1)
            f = f.permute(0, 3, 1, 2).contiguous()
            # upsample each layer to same target_hw before fusion
            f = self.reassembles[i](f, target_hw)
            projected.append(f)

        logits = self.head(projected)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return logits