from __future__ import print_function

import json
import os
from collections import defaultdict

import numpy as np
import torch
from adabound import AdaBound
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils import data

import environments
from callback import TensorboardLogger, LoggingCallback, CallbackManager, SlackNotifyCallback, TestGenerateCallback
from config import Config
from dataset import DatasetFromQuery
from model import SRCNN, BNSRCNN
from utils.logger import get_logger
from utils.common import class_to_dict
from validation import psnr_score, Set5Validation

logger = get_logger(__name__, output_file=os.path.join(Config.checkpoints_path, 'log.txt'))

DATASETS = {
    '91': DatasetFromQuery(query=os.path.join(environments.DATASET_DIR, 'train_91/*.bmp')),
    'Set5': DatasetFromQuery(query=os.path.join(environments.DATASET_DIR, 'Set5/*.bmp'))
}

MODELS = {
    'srcnn': SRCNN,
    'bnsrcnn': BNSRCNN
}


def save_model(model, save_path, name, iter_cnt):
    os.makedirs(save_path, exist_ok=True)

    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    logger.info(f'save {name} to {save_name}')
    return save_name


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def calculate_metrics(output, label) -> dict:
    data = {}
    mse_loss = nn.MSELoss()(output, label).item()
    data['psnr'] = psnr_score(mse_loss)
    return data


class ClopTensor:
    def __init__(self, center=(32, 32), size=40):
        self.center = center
        self.size = size

        self.left = int(center[0] - size / 2)
        self.right = self.left + size
        self.top = int(center[1] - size / 2)
        self.bottom = self.left + size

    def call(self, x):
        return x[:, :, self.left:self.right, self.top:self.bottom]


class ClopLoss:
    def __init__(self):
        self.clipping = ClopTensor()
        self.loss = nn.MSELoss()

    def __call__(self, x, y):
        x = self.clipping.call(x)
        y = self.clipping.call(y)
        return self.loss(x, y)


LOSSES = {
    'mse': nn.MSELoss(),
    'mse-clop': ClopLoss()
}

if __name__ == '__main__':

    opt = Config()
    device = torch.device("cuda")

    train_loader = data.DataLoader(DATASETS.get(Config.dataset, None),
                                   batch_size=opt.train_batch_size,
                                   shuffle=True,
                                   num_workers=opt.num_workers)

    logger.info('{} train iters per epoch:'.format(len(train_loader)))
    model = MODELS.get(Config.model, None)()
    logger.info(model)
    model.to(device)
    criterion = LOSSES.get(Config.loss, None)

    params = [{'params': model.parameters()}]
    if Config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr=opt.lr, weight_decay=opt.weight_decay, momentum=.9,
                                    nesterov=True)
    elif Config.optimizer == 'adabound':
        optimizer = AdaBound(params=params,
                             weight_decay=opt.weight_decay,
                             lr=opt.lr,
                             final_lr=opt.final_lr,
                             amsbound=opt.amsbound)
    elif Config.optimizer == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=opt.lr,
                                     weight_decay=opt.weight_decay)
    else:
        raise ValueError('Invalid Optimizer Name: {}'.format(Config.optimizer))
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    callback_manager = CallbackManager([
        TensorboardLogger(log_dir=Config.checkpoints_path),
        LoggingCallback(),
        TestGenerateCallback(query=os.path.join(environments.DATASET_DIR, 'Set5/*.bmp'),
                             model=model,
                             output=os.path.join(Config.checkpoints_path, 'Set5'))
    ])

    validations = [
        Set5Validation(dirpath=os.path.join(environments.DATASET_DIR, 'Set5'))
    ]

    if environments.SLACK_INCOMING_URL and not Config.is_debug:
        logger.info('Add Slack Notification')
        callback_manager.callbacks.append(SlackNotifyCallback(url=environments.SLACK_INCOMING_URL, config=Config))

    with open(Config.config_path, 'w') as f:
        data = class_to_dict(Config)
        json.dump(data, f, indent=4, sort_keys=True)
        logger.info(f'save config to {Config.config_path}')

    try:
        for epoch in range(opt.max_epoch):
            scheduler.step()
            model.train()
            callback_manager.on_epoch_start(epoch)

            for i, data in enumerate(train_loader):
                callback_manager.on_batch_start(n_batch=i)
                data_input, label = data
                data_input = data_input.to(device)
                label = label.to(device)
                output = model(data_input)
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iters = epoch * len(train_loader) + i

                metric = calculate_metrics(output, label)
                metric['loss'] = loss.item()
                metric['lr'] = get_lr(optimizer)
                callback_manager.on_batch_end(loss=loss.item(), n_batch=i, train_metric=metric)
                if Config.is_debug:
                    break
            if epoch % opt.save_interval == 0 or epoch == opt.max_epoch:
                save_model(model, opt.checkpoints_path, 'generator', epoch)

            model.eval()
            valid_metrics = {}
            for v in validations:
                m = v.call(model)
                valid_metrics.update(m)

            callback_manager.on_epoch_end(epoch, valid_metric=valid_metrics)

        callback_manager.on_end_train()
    except KeyboardInterrupt as e:
        callback_manager.on_end_train(e)

    except Exception as e:
        import traceback

        logger.warning(traceback.format_exc())
        callback_manager.on_end_train(e)
