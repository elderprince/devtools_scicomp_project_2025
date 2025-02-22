import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # 1st conv layer: 96 x 55 x 55
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            # 1st maxpool: 96 x 27 x 27
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 2nd conv layer: 256 x 27 x 27
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), 
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            # 2nd maxpool: 256 x 13 x 13
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 3rd conv layer: 384 x 13 x 13
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            # 4th conv layer: 384 x 13 x 13
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            # 5th conv layer: 256 x 13 x 13
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            # 1st linear layer
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            # 2nd linear layer
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            # classifier layer
            nn.Linear(in_features=4096, out_features=num_classes),
        )
    def forward(self, x):
        x = self.avgpool(self.features(x)).flatten(start_dim=1)
        logits = self.classifier(x)
        return logits