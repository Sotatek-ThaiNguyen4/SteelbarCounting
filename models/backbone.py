from torch import nn
import torchvision


def _build_torchvision_resnet50(pretrained=True):
    try:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        return torchvision.models.resnet50(weights=weights)
    except AttributeError:
        return torchvision.models.resnet50(pretrained=pretrained)


BACKBONE_REGISTRY = {
    "resnet50": {
        "builder": _build_torchvision_resnet50,
        "pyramid_channels": (256, 512, 1024),
    },
}


def get_supported_backbones():
    return tuple(BACKBONE_REGISTRY.keys())


class BackboneResNet(nn.Module):
    def __init__(self, name: str, return_interm_layers: bool):
        super().__init__()
        config = BACKBONE_REGISTRY[name]
        self.backbone = config["builder"](pretrained=True)
        self.return_interm_layers = return_interm_layers
        self.num_channels = config["pyramid_channels"][0]
        self.pyramid_channels = config["pyramid_channels"]

    def forward(self, x):
        out = []

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        out.append(x)

        x = self.backbone.layer1(x)
        out.append(x)
        x = self.backbone.layer2(x)
        out.append(x)
        x = self.backbone.layer3(x)
        out.append(x)

        if self.return_interm_layers:
            return out
        return [out[-1]]


def build_backbone(args):
    name = args.backbone.lower()
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unsupported backbone '{args.backbone}'. "
            f"Available: {', '.join(get_supported_backbones())}"
        )
    return BackboneResNet(name, True)
