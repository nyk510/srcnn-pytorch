import torch
from PIL import Image
from torchvision import transforms as T
from utils.common import get_model_device


def make_reconstruction_image(model, img, shrink_ratio=.5, interpolation=Image.BICUBIC):
    min_size = [int(x * shrink_ratio) for x in img.size]
    input_transform = T.Compose([
        T.Resize(size=min_size[::-1]),
        T.Resize(size=img.size[::-1], interpolation=interpolation),
        T.ToTensor()
    ])

    x = input_transform(img)
    device = get_model_device(model)
    x = x.to(device)

    with torch.no_grad():
        pred = model(x[None, :, :, :])
    return pred[0], x
