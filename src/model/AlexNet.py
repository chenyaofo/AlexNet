import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, n_classes):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # 3@227*227
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            # 96@55*55
            nn.ReLU(inplace=True),
            # 96@55*55
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 96@27*27
            nn.BatchNorm2d(num_features=96),

            # 96@27*27
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            # 256@27*27
            nn.ReLU(inplace=True),
            # 256@27*27
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 256@13*13
            nn.BatchNorm2d(num_features=256),

            # 256@13*13
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            # 384@13*13
            nn.ReLU(inplace=True),

            # 384@13*13
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            # 384@13*13
            nn.ReLU(inplace=True),

            # 384@13*13
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 256@13*13
            nn.ReLU(inplace=True),
            # 256@13*13
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 256@6*6
        )

        self.classifier = nn.Sequential(
            # 256@6*6
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            # 4096
            nn.ReLU(inplace=True),
            # 4096
            nn.Dropout(p=0.5),

            # 4096
            nn.Linear(in_features=4096, out_features=4096),
            # 4096
            nn.ReLU(inplace=True),
            # 4096
            nn.Dropout(p=0.5),

            # 4096
            nn.Linear(in_features=4096, out_features=n_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(-1, 256 * 6 * 6)
        out = self.classifier(out)
        return out
