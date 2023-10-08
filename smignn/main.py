import argparse
import os
import torch
from save_config import My_config, update_default_config
from allutils.data import Data
from model.trainer import Trainer
from allutils.utils import set_random_seed

def initialize(seed=0):
    set_random_seed(seed)

def update_config(conf):
    if conf.data_name == 'yelp':
        conf.num_users = 17237
        conf.num_items = 38342
    elif conf.data_name == 'ciao':
        conf.num_users = 7375
        conf.num_items = 106797
    else:
        raise NotImplementedError('error')

    update_default_config(conf)
    conf.cur_out_path = os.path.join(conf.out_path, conf.data_name + str(conf.train_ratio) + str(conf.conf_name))
    os.makedirs(conf.cur_out_path, exist_ok=True)
    conf.log_path = os.path.join(conf.cur_out_path, 'log.txt')
    conf.result_path = os.path.join(conf.cur_out_path, 'result.txt')


def batch_test_model(conf):
    train_ratio_list = [0.2, 0.4, 0.6, 0.8]
    for train_ratio in train_ratio_list:
        conf.train_ratio = train_ratio
        update_config(conf)
        dataset = Data(conf)
        trainer = Trainer(conf, dataset)
        trainer.test()


def batch_train_model(conf):
    train_ratio_list = [0.2, 0.4, 0.6, 0.8]
    for train_ratio in train_ratio_list:
        conf.train_ratio = train_ratio
        update_config(conf)
        dataset = Data(conf)
        trainer = Trainer(conf, dataset)
        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment setup')
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--data_name', default='yelp', type=str)
    parser.add_argument('--out_path', default='./output', type=str)
    parser.add_argument('--conf_name', default='smignn', type=str)

    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--num_train_negatives', default=10, type=int)
    parser.add_argument('--num_eval_negatives', default=1000, type=int)
    parser.add_argument('--num_test_negatives', default=1000, type=int)
    parser.add_argument('--top_k', default=20, type=int)
    parser.add_argument('--num_proc', default=4, type=int)

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--train_batch_size', default=512, type=int)
    parser.add_argument('--test_batch_size', default=512, type=int)
    parser.add_argument('--eval_epochs', default=1, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--patience', default=30, type=int)
    parser.add_argument('--loss', default='cross', type=str)

    parser.add_argument('--extra_message', default='', type=str)
    parser.add_argument('--train_ratio', default=0.8, type=float)
    parser.add_argument('--train_model', default='true')
    parser.add_argument('--test_model', default='true')
    parser.add_argument('--batch_train_model', action='store_true')
    parser.add_argument('--batch_test_model', action='store_true')

    args = vars(parser.parse_args())
    conf = My_config(args)

    initialize(conf.seed)
    os.makedirs(conf.out_path, exist_ok=True)
    conf.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(conf.__dict__)

    if conf.batch_test_model:
        print('batch_test_model')
        batch_test_model(conf)
    if conf.batch_train_model:
        print('batch_train_model')
        batch_train_model(conf)
    update_config(conf)
    if conf.train_model:
        conf.save()
    dataset = Data(conf)
    trainer = Trainer(conf, dataset)
    if conf.train_model:
        trainer.train()
    if conf.test_model:
        trainer.test()