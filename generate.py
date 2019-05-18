import os
from argparse import ArgumentParser

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image

from model import SRCNN


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('-w', '--weight', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--scale', type=float, default=2.)
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = get_arguments()

    weight = args.get('weight')
    p = args.get('input')
    upscale = args.get('scale')

    dirpath = os.path.dirname(p)
    input_filename = os.path.basename(p)

    gen = SRCNN()
    gen.load_state_dict(torch.load(weight))
    gen.eval()

    img = Image.open(p)
    new_size = [int(x * upscale) for x in img.size]

    converter = T.Compose([
        T.Resize(size=new_size[::-1], interpolation=Image.BICUBIC),
        T.ToTensor()
    ])

    with torch.no_grad():
        x = converter(img)
        pred = gen(x[None, :, :, :])[0]

    save_image(pred, os.path.join(dirpath, f'outputx{upscale}_{input_filename}'))
