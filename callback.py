"""
学習時の epoch 開始/終了などをトリガに動くコールバック関数の定義
"""

import json
import os
from collections import defaultdict
from glob import glob
from typing import List

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tensorboardX import SummaryWriter
from torchvision.utils import save_image

from test import make_reconstruction_image
from utils.logger import get_logger

logger = get_logger(__name__)


class AbstractCallback:
    def on_epoch_start(self, epoch: int):
        pass

    def on_epoch_end(self, epoch: int, valid_metric: dict):
        pass

    def on_batch_start(self, n_batch, *args, **kwargs):
        pass

    def on_batch_end(self, loss, n_batch, train_metric: dict):
        pass

    def on_end_train(self, exception=None):
        pass


class TensorboardLogger(AbstractCallback):
    """
    Tensorboard への log の記録 + csv で epoch ごとの値を保存する callback
    """

    def __init__(self, log_dir=None):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.current_epoch = -1
        self.history_df = pd.DataFrame()

    def on_epoch_start(self, epoch):
        self.current_epoch = epoch
        self.epoch_score_df = pd.DataFrame()

    def on_batch_end(self, loss, n_batch, train_metric: dict):
        df_i = pd.DataFrame([train_metric])
        df_i['loss'] = loss
        df_i['epoch'] = self.current_epoch
        df_i['n_batch'] = n_batch

        self.epoch_score_df = pd.concat([self.epoch_score_df, df_i], ignore_index=True)

    def on_epoch_end(self, epoch, valid_metric: dict):
        df = self.epoch_score_df.drop(columns=['epoch', 'n_batch'])
        train_metric = df.mean().to_dict()

        for k, v in train_metric.items():
            self.writer.add_scalar(f'train_{k}', scalar_value=v, global_step=epoch)
        for k, v in valid_metric.items():
            self.writer.add_scalar(f'valid_{k}', scalar_value=v, global_step=epoch)

        data = self.epoch_score_df.mean()
        data = data.add_prefix('train_')
        for k, v in valid_metric.items():
            data[f'valid_{k}'] = v

        self.history_df = self.history_df.append(data, ignore_index=True)
        self.history_df.to_csv(os.path.join(self.writer.log_dir, 'train_log.csv'), index=False)


def get_str_from_dict(d):
    s = []
    for k, v in d.items():
        if v > 1:
            s.append(f'{k}: {v:.3f}')
        else:
            s.append(f'{k}: {v:.3e}')

    s = ' '.join(s)
    return s


class SlackNotifyCallback(AbstractCallback):
    def __init__(self, url, config):
        self.url = url
        self.epoch_score_df = pd.DataFrame()

        def props(cls):
            return [[i, getattr(cls, i)] for i in cls.__dict__.keys() if i[:1] != '_']

        self.conf_data = dict(props(config))

    def on_epoch_start(self, epoch):
        self.current_epoch = epoch

        if epoch > 0:
            return

        self.post(text=json.dumps(self.conf_data, indent=4), title='Start Training')

    def on_batch_end(self, loss, n_batch, train_metric: dict):
        df_i = pd.DataFrame([train_metric])
        df_i['loss'] = loss
        df_i['epoch'] = self.current_epoch

        self.epoch_score_df = pd.concat([self.epoch_score_df, df_i], ignore_index=True)

    def on_epoch_end(self, epoch, valid_metric: dict):
        df = self.epoch_score_df[self.epoch_score_df['epoch'] == self.current_epoch]
        train_metric = df.mean().to_dict()

        text = f'[epoch: {epoch:03d}] train: {get_str_from_dict(train_metric)} valid: {get_str_from_dict(valid_metric)}'
        self.post(text=text)

    def post(self, text, title=None, color=None):
        payload = {
            'attachments': [{
                'text': text,
                'title': title,
                'color': color
            }]
        }
        requests.post(self.url, json.dumps(payload))

    def on_end_train(self, exception=None):
        if exception is None:
            return

        import traceback

        if isinstance(exception, KeyboardInterrupt):
            self.post(text='keyboard interrupted.')
            return

        self.post(text=traceback.format_exc(), title='Training Is Stopped', color='#d80b65')


class LoggingCallback(AbstractCallback):
    def __init__(self, log_freq=100):
        self.log_freq = log_freq
        self.epoch = 0
        self._init()

    def on_epoch_start(self, epoch: int):
        self.epoch = epoch

    def on_batch_end(self, loss, n_batch, train_metric: dict):
        for k, v in train_metric.items():
            self.losses[k].append(v)

        if n_batch % self.log_freq != 0:
            return

        data = dict([[k, np.mean(v)] for k, v in self.losses.items()])
        print(f'[epoch:{self.epoch:04d}-{n_batch:05d}] {get_str_from_dict(data)}')

    def _init(self):
        self.losses = defaultdict(list)

    def on_epoch_end(self, epoch: int, valid_metric: dict):
        s = get_str_from_dict(valid_metric)
        print(f'[validate] {s}')
        self._init()


class TestGenerateCallback(AbstractCallback):
    """
    与えられた glob query に該当する画像に対して, epoch ごとに高解像度化を行う callback
    model への入力画像は元画像を `shrink` 倍に縮小して, BICUBIC によって元の画像サイズに戻されたものを用います.
    詳しくは `test.py` を参照のこと
    """

    def __init__(self, query, model, output, shrink=.3):
        self.img_paths = glob(query)
        self.output = output
        assert len(self.img_paths) > 0
        self.model = model
        self.shrink = shrink

    def on_epoch_start(self, epoch: int):
        os.makedirs(self.output, exist_ok=True)

    def on_epoch_end(self, epoch: int, valid_metric: dict):
        for p in self.img_paths:
            img = Image.open(p)
            x, input_tensor = make_reconstruction_image(self.model, img, shrink_ratio=self.shrink)
            base_name = os.path.basename(p).split('.')[0]
            fname = f'{base_name}_epoch={epoch}.png'
            save_image(x, os.path.join(self.output, fname))


class CallbackManager(AbstractCallback):
    def __init__(self, callbacks: List[AbstractCallback]):
        self.callbacks = callbacks

    def on_epoch_start(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_epoch_start(*args, **kwargs)

    def on_epoch_end(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_epoch_end(*args, **kwargs)

    def on_batch_start(self, *args, **kwargs):
        for c in self.callbacks:
            c.on_batch_start(*args, **kwargs)

    def on_batch_end(self, loss, n_batch, train_metric: dict):
        for c in self.callbacks:
            c.on_batch_end(loss, n_batch, train_metric)

    def on_end_train(self, exception=None):
        for c in self.callbacks:
            c.on_end_train(exception)
