from torchvision import transforms as T
from glob import glob
import os
from PIL import Image
from utils.common import get_model_device
from torch import nn
import numpy as np
import torch
from model import ESPCN, generate_high_img


def psnr_score(mse_loss):
    return 10 * np.log10(1 / mse_loss)


def calculate_valid_crop_size(crop_size, upscale_factor):
    x = crop_size - (crop_size % upscale_factor)
    return x, int(x / upscale_factor)

class AbstractValidation:
    name = None

    def call_core(self, model):
        raise NotImplementedError()

    def call(self, model):
        metric = self.call_core(model)
        m = {}
        for k, v in metric.items():
            m[f'{self.name}_{k}'] = v
        return m


class Set5Validation(AbstractValidation):
    name = 'set5'

    def __init__(self, dirpath, shrink_ratio=2):
        self.dirpath = dirpath
        self.low_dirname = f'lowx{shrink_ratio}'

        self.original_imgs = glob(os.path.join(dirpath, '*.bmp'))
        self.low_imgs = glob(os.path.join(dirpath, self.low_dirname, '*.bmp'))
        assert len(self.original_imgs) > 0
        assert len(self.low_imgs) > 0
        self.mse_loss = nn.MSELoss()

    def get_low_img(self, name):
        for p in self.low_imgs:
            if name in p:
                return p
        return None

    def iter_images(self):
        for origin in self.original_imgs:
            origin_img = Image.open(origin)
            name = os.path.basename(origin)
            low_img = Image.open(self.get_low_img(name))
            yield origin_img, low_img

    def call_core(self, model):
        model.eval()
        device = get_model_device(model)

        def converter(img):
            x = T.ToTensor()(img)
            x = x[None, :, :, :]
            x = x.to(device)
            return x

        losses = []
        for origin, low in self.iter_images():

            t = converter(origin)
            with torch.no_grad():
                if isinstance(model, ESPCN):
                    low = origin.resize([int(i / model.upscale) for i in low.size])
                    y = generate_high_img(model, low_img=low, as_tensor=True, device=device)
                else:
                    x = converter(low)
                    y = model(x)
                print(y.shape, t.shape)
                loss = self.mse_loss(t, y)
            losses.append(loss.item())
        losses = np.array(losses)

        metrics = {}
        metrics['mse'] = losses.mean()
        metrics['psnr'] = psnr_score(losses).mean()
        return metrics
