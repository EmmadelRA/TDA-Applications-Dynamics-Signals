import torch
import torch.nn as nn

class Tiny2DBackbone(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 5, kernel_size=3, dilation=2, padding=2, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3, dilation=4, padding=4, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.gap(x).view(x.size(0), -1)
        return x

class PersistenceCNN(nn.Module):
    def __init__(self, input_channels=2, num_classes=2):
        super().__init__()
        self.backbone = Tiny2DBackbone(in_channels=input_channels)
        self.fc = nn.Linear(10, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.fc(feats)
        return logits
