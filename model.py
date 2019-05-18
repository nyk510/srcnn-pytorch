from torch import nn


class SRCNN(nn.Module):
    """
    SRCNN

    ref: http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
    """

    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.activate(self.conv1(x))
        h = self.activate(self.conv2(h))
        return self.conv3(h)


class BNSRCNN(nn.Module):
    """
    SRCNN with BatchNormalization
    """

    def __init__(self):
        super(BNSRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.activate(self.bn1(self.conv1(x)))
        h = self.activate(self.bn2(self.conv2(h)))
        return self.conv3(h)
