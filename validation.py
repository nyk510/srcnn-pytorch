from torchvision import transforms as T
from glob import glob
import os
from PIL import Image
from utils.common import get_model_device, calculate_original_img_size
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
    """
    Validation on Set5 Dataset

    original imgs の glob さえ書き換えればどんなデータセットでも行けるので
    Image Validation みたいな汎用クラスにまとめたい
    """

    name = 'set5'

    def __init__(self, dirpath, upscale=2):
        self.dirpath = dirpath
        self.dirname = f'scalex{upscale}'
        self.img_dir = os.path.join(dirpath, self.dirname)

        self.upscale = upscale
        self.img_set_list = []

        self.original_imgs = glob(os.path.join(dirpath, '*.bmp'))
        assert len(self.original_imgs) > 0

        self.mse_loss = nn.MSELoss()
        self.make_low_and_hig_resolution_images()

    def make_low_and_hig_resolution_images(self):
        os.makedirs(self.img_dir, exist_ok=True)
        for p in self.original_imgs:
            name = os.path.basename(p)
            img = Image.open(p)
            high_size = [calculate_original_img_size(i, self.upscale) for i in img.size]
            low_size = [int(i / self.upscale) for i in high_size]
            high_img = img.crop((0, 0, *high_size))
            low_img = high_img.resize(low_size, Image.BICUBIC)
            high_img.save(os.path.join(self.img_dir, f'high_{name}'))
            low_img.save(os.path.join(self.img_dir, f'low_{name}'))

            self.img_set_list.append([high_img, low_img])

    def call_core(self, model):
        model.eval()
        device = get_model_device(model)

        def converter(img):
            x = T.ToTensor()(img)
            x = x[None, :, :, :]
            x = x.to(device)
            return x

        losses = []
        for origin, low in self.img_set_list:

            t = converter(origin)
            with torch.no_grad():
                if isinstance(model, ESPCN):
                    y = generate_high_img(model, low_img=low, as_tensor=True, device=device)
                else:
                    low = low.resize(origin.size, Image.BICUBIC)
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
