from glob import glob

from PIL import Image
from torch.utils import data
from torchvision import transforms as T

from utils.common import calculate_original_img_size


def get_luminance(img):
    x, _, _ = img.convert('YCbCr').split()
    return x


class DatasetFromQuery(data.Dataset):
    def __init__(self,
                 query,
                 max_size=128,
                 shrink_scale=2,
                 total_samples=100000,
                 only_luminance=False,
                 input_upsample=True,
                 interpolation=Image.BICUBIC):
        """
        Glob Query にマッチする画像からランダムに切り取られた正方形の画像と
        それを縮小した画像のペアを返すデータセット

        引数によって既に縮小された画像を返すかどうかが変わってくるので注意

        Args:
            query: glob query
            max_size:
                ランダムに切り取る画像の最大 pixel 数.
                正確な値は shrink scale とともに `calculate_original_img_size` によって計算される値になります.
            shrink_scale:
                切り取られた画像を縮小するスケール.
                例えば 3 が設定されると, 切り取られた画像を 1/3 に縮小します.
            total_samples:
                1 epoch あたりの画像枚数. 指定しないとすべての画像を一通り見ると 1epoch になります.
                特定の値を設定すると, その数だけランダムにデータを pickup したときに 1epoch とします.
            only_luminance:
                True の時輝度情報のレイヤのみを返します. すなわち channel=1 の画像となります.
            input_upsample:
                True のとき入力画像を resize して target 画像と同じ大きさに前もって変換します.
                変換方法は `interpolation` で指定されたアルゴリズムを使用します.
            interpolation:
        """

        high_size = calculate_original_img_size(max_size, upscale_factor=shrink_scale)
        low_size = int(high_size / shrink_scale)

        self.low_size = low_size
        self.high_size = high_size

        super(DatasetFromQuery, self).__init__()
        self.img_paths = list(glob(query))
        self.only_luminance = only_luminance

        if total_samples:
            self.total_samples = total_samples
        else:
            self.total_samples = len(self.img_paths)

        self.n_images = len(self.img_paths)
        self.to_tensor = T.ToTensor()

        if input_upsample:
            self.input_transform = T.Compose([
                T.Resize(size=low_size),
                T.Resize(size=high_size, interpolation=interpolation),
            ])
        else:
            self.input_transform = T.Compose([
                T.Resize(size=low_size, interpolation=interpolation),
            ])

        self.preprocess = T.Compose([
            T.RandomCrop(size=high_size),
            T.RandomHorizontalFlip()
        ])

    def __len__(self):
        return self.total_samples

    def __getitem__(self, i):
        img = Image.open(self.img_paths[i % self.n_images])
        img = self.preprocess(img)
        img_target = img.copy()
        img_input = self.input_transform(img)

        if self.only_luminance:
            img_input = get_luminance(img_input)
            img_target = get_luminance(img_target)

        return self.to_tensor(img_input), self.to_tensor(img_target)
