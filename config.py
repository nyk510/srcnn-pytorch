import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime

import environments


def get_time_string(t=None, formatting='{0:%Y-%m-%d_%H-%M-%S}'):
    if t is None:
        t = datetime.now()
    return formatting.format(t)


def get_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description=__doc__)

    parser.add_argument('--debug', action='store_true',
                        help='If add it, run with debugging mode (no record and stop one batch per epoch')
    # model setting
    parser.add_argument('--model', type=str, default='srcnn', help='model architecture name')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function', choices=['mse', 'mse-clop'])

    # dataset setting
    parser.add_argument('--dataset', type=str, default='91', help='dataset name')
    parser.add_argument('--valid', type=str, default='Set5', help='validation dataset name')

    # optimizer settings
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer Name')
    parser.add_argument('--lr', type=float, default=.1, help='learning rate')
    parser.add_argument('--decay', type=float, default=1e-8, help='weight decay')
    parser.add_argument('--final_lr', type=float, default=.1,
                        help='final learning rate (only activa on `optimizer="adabound"`')

    # run settings
    parser.add_argument('--batch', type=int, default=128, help='training batch size')
    return vars(parser.parse_args())


class Config(object):
    """
    学習/検証を実行する際の設定値
    """
    args = get_arguments()
    now = get_time_string()

    is_debug = args.get('debug', False)

    model = args.get('model', None)
    loss = args.get('loss', None)
    dataset = args.get('dataset', None)
    valid_dataset = args.get('valid', None)

    save_interval = 5
    train_batch_size = args.get('batch', 32)  # batch size
    valid_batch_size = 64
    # optimizer name: `"adam"`, `"sgd"`, `"adabound"`
    optimizer = args.get('optimizer', None)
    num_workers = 4  # how many workers for loading data

    max_epoch = 30
    lr = args.get('lr', 0.1)  # initial learning rate
    lr_step = 10  # cut lr frequency
    weight_decay = args.get('decay')

    # use in adabound
    final_lr = args.get('final_lr', .2)
    amsbound = True

    checkpoints_path = os.path.join(environments.DATASET_DIR, 'checkpoints',
                                    f'{dataset}_{model}_{loss}_{optimizer}_lr={lr:.1e}_final={final_lr:.1e}_{now}')
    config_path = os.path.join(checkpoints_path, 'train_config.json')


print(Config.checkpoints_path)
os.makedirs(Config.checkpoints_path, exist_ok=True)
