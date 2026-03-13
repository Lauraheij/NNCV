import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Semantic segmentation model using a DINOv2 Vision Transformer backbone
    with a linear segmentation head.

    DINOv2 (Oquab et al., 2023) is a self-supervised ViT pretrained on 142M images,
    providing strong general-purpose visual features. We attach a lightweight linear
    head that upsamples patch tokens back to the full resolution segmentation mask.

    Reference: https://arxiv.org/abs/2304.07193

    Args:
        in_channels (int): Number of input channels. Default is 3 for RGB.
        n_classes (int): Number of output segmentation classes. Default is 19 (Cityscapes).
        backbone (str): DINOv2 variant to use. Options:
            - 'dinov2_vits14': Small  (~22M params, fastest)
            - 'dinov2_vitb14': Base   (~86M params, best accuracy)
            - 'dinov2_vitl14': Large  (~307M params, very slow)
    """

    def __init__(
        self,
        in_channels=3,
        n_classes=19,
        backbone='dinov2_vitb14',
    ):
        super().__init__()

        self.n_classes = n_classes
        self.patch_size = 14  # DINOv2 uses 14x14 patches

        # Load pretrained DINOv2 backbone from torch.hub
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            backbone,
            pretrained=True,
        )

        # Freeze all backbone parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last 2 transformer blocks
        for param in self.backbone.blocks[-2:].parameters():
            param.requires_grad = True

        # Unfreeze the final norm layer
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

        # Get the embedding dimension for the chosen backbone variant
        embed_dims = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
        }
        embed_dim = embed_dims[backbone]

        # Linear segmentation head: maps patch embeddings → class logits
        # Simple but effective — justified by the strength of DINOv2 features
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            logits: Tensor of shape (B, n_classes, H, W)
        """
        B, C, H, W = x.shape

        # Get patch tokens from DINOv2 (excludes [CLS] token)
        # Output shape: (B, num_patches, embed_dim)
        features = self.backbone.get_intermediate_layers(x, n=1)[0]

        # Calculate feature map spatial dimensions
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        # Reshape from (B, num_patches, embed_dim) → (B, embed_dim, h_patches, w_patches)
        features = features.reshape(B, h_patches, w_patches, -1)
        features = features.permute(0, 3, 1, 2).contiguous()

        # Apply segmentation head
        logits = self.head(features)

        # Upsample back to input resolution (B, n_classes, H, W)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        return logits