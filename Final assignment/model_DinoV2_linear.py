import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class LinearHead(nn.Module):
    def __init__(self, embed_dim=768, n_classes=19):
        super().__init__()
        self.classifier = nn.Conv2d(embed_dim, n_classes, kernel_size=1)

    def forward(self, x):
        return self.classifier(x)


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

        # Frozen part — same as peak model
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last 6 blocks — same as peak model
        for block in self.backbone.blocks[-6:]:
            for param in block.parameters():
                param.requires_grad = True

        # Always unfreeze norm — same as peak model
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

        embed_dims = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
        }
        embed_dim = embed_dims[backbone]

        # CHANGE: simple linear head instead of DPT
        self.head = LinearHead(embed_dim=embed_dim, n_classes=n_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        # CHANGE: only use last layer — no multi-scale
        feat = self.backbone.get_intermediate_layers(x, n=[11])[0]
        feat = feat.reshape(B, h_patches, w_patches, -1)
        feat = feat.permute(0, 3, 1, 2).contiguous()  # (B, embed_dim, h_p, w_p)

        logits = self.head(feat)  # (B, n_classes, h_p, w_p)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return logits