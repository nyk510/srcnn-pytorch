from glob import glob

from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class DatasetFromQuery(data.Dataset):
    def __init__(self,
                 query,
                 min_size=128,
                 shrink_scale=2,
                 total_samples=100000,
                 only_luminance=False,
                 input_upsample=True,
                 interpolation=Image.BICUBIC):

        min_size = min_size
        high_size = min_size * shrink_scale
        super(DatasetFromQuery, self).__init__()
        self.img_paths = list(glob(query))
        self.only_luminance = only_luminance
        self.total_samples = total_samples
        self.n_images = len(self.img_paths)

        if input_upsample:
            self.input_transform = T.Compose([
                T.Resize(size=min_size),
                T.Resize(size=high_size, interpolation=interpolation),
                T.ToTensor()
            ])
        else:
            self.input_transform = T.Compose([
                T.Resize(size=min_size),
                T.ToTensor()
            ])

        self.target_transform = T.Compose([
            T.ToTensor()
        ])

        self.preprocess = T.Compose([
            T.RandomCrop(size=high_size)
        ])

    def __len__(self):
        return self.total_samples

    def __getitem__(self, i):
        img = Image.open(self.img_paths[i % self.n_images])
        if self.only_luminance:
            img, _, _ = img.convert('YCbCr').split()

        img = self.preprocess(img)
        target = img.copy()

        img = self.input_transform(img)
        target = self.target_transform(target)
        return img, target
