from glob import glob

from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class DatasetFromQuery(data.Dataset):
    def __init__(self,
                 query,
                 size=64,
                 min_size=32,
                 total_samples=100000,
                 interpolation=Image.BICUBIC):
        super(DatasetFromQuery, self).__init__()
        self.img_paths = list(glob(query))
        self.total_samples = total_samples
        self.n_images = len(self.img_paths)

        self.input_transform = T.Compose([
            T.Resize(size=min_size),
            T.Resize(size=size, interpolation=interpolation),
            T.ToTensor()
        ])
        self.target_transform = T.Compose([
            T.ToTensor()
        ])

        self.preprocess = T.Compose([
            T.RandomRotation(degrees=90),
            T.RandomCrop(size=size)
        ])

    def __len__(self):
        return self.total_samples

    def __getitem__(self, i):
        img = Image.open(self.img_paths[i % self.n_images])
        img = self.preprocess(img)
        target = img.copy()

        img = self.input_transform(img)
        target = self.target_transform(target)
        return img, target
