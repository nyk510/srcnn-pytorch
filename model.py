from torch import nn
import torchvision.transforms as T
from PIL import Image
import torch
import numpy as np


class AbstractNet(nn.Module):
    only_luminance = False
    input_upscale = True


class SRCNN(AbstractNet):
    """
    SRCNN

    ref: http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
    """

    def __init__(self, **kwargs):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.activate(self.conv1(x))
        h = self.activate(self.conv2(h))
        return self.conv3(h)


class BNSRCNN(AbstractNet):
    """
    SRCNN with BatchNormalization
    """

    def __init__(self, **kwargs):
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


class ESPCN(AbstractNet):
    only_luminance = True
    input_upscale = False

    def __init__(self, upscale=2):
        super(ESPCN, self).__init__()
        self.upscale = upscale
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, upscale ** 2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(self.upscale)

        self.relu = nn.ReLU()

    def _initialize_weights(self):
        weights_with_relu = [
            self.conv1.weight,
            self.conv2.weight,
            self.conv3.weight
        ]

        for w in weights_with_relu:
            nn.init.orthogonal_(w, nn.init.calculate_gain('relu'))

        nn.init.orthogonal_(self.conv4.weight)

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        return self.pixel_shuffle(self.conv4(h))


MODELS = {
    'srcnn': SRCNN,
    'bnsrcnn': BNSRCNN,
    'espcn': ESPCN
}


def get_network(name):
    for n, cls in MODELS.items():
        if n == name:
            return cls

    raise ValueError(f'Model {name} is not defined. ')


def to_pil_image(x):
    out_img_y = x.detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    return out_img_y


def generate_high_img(network, low_img, as_tensor=False, device=None):
    x, y, z = low_img.convert('YCbCr').split()
    with torch.no_grad():
        x = T.ToTensor()(x)[None, :, :, :]
        x = x.to(device)
        x_high = network(x)[0]
        x_high = x_high.cpu()

    x_high = to_pil_image(x_high)
    img = Image.merge('YCbCr', [x_high, y.resize(x_high.size), z.resize(x_high.size)]).convert('RGB')
    if as_tensor:
        return T.ToTensor()(img).to(device)
    return img
