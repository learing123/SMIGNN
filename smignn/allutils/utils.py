import numpy as np
import random
import torch
import dgl
from collections import defaultdict

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_source_target_form_pair_set(pair_set):
    source_ids = []
    target_ids = []
    for id_1, id_2 in pair_set:
        source_ids.append(id_1)
        target_ids.append(id_2)
    return source_ids, target_ids


def reverse_pair_set(pair_set):
    rev_pair_set = set()
    for (id_1, id_2) in pair_set:
        rev_pair_set.add((id_2, id_1))
    return rev_pair_set


def reverse_pair_list(pair_list):
    rev_pair_list = list()
    for (id_1, id_2) in pair_list:
        rev_pair_list.append((id_2, id_1))
    return rev_pair_list


def construct_hetero_graphs(pair_sets, edge_types, num_nodes_dict):
    data_dict = {}
    for idx, pair_set in enumerate(pair_sets):
        source_ids, target_ids = get_source_target_form_pair_set(pair_set)
        data_dict[edge_types[idx]] = (source_ids, target_ids)
    graph = dgl.heterograph(data_dict, num_nodes_dict)
    return graph

def split_pair_dict_random_ratio(pair_set, test_ratio, valid_ratio, train_ratio):
    set_random_seed(2023)
    assert test_ratio + valid_ratio + train_ratio <= 1.0
    train_dict = defaultdict(set)
    valid_dict = defaultdict(set)
    test_dict = defaultdict(set)
    for user, item in pair_set:
        thr_0 = test_ratio
        thr_1 = test_ratio + valid_ratio
        thr_2 = test_ratio + valid_ratio + train_ratio

        rand_val = random.uniform(0.0, 1.0)
        if 0 <= rand_val < thr_0:
            test_dict[user].add(item)
        elif thr_0 <= rand_val < thr_1:
            valid_dict[user].add(item)
        elif thr_1 <= rand_val < thr_2:
            train_dict[user].add(item)

    return train_dict, valid_dict, test_dict


def get_pair_set_from_pair_dict(pair_dict):
    pair_set = set()
    for key, val in pair_dict.items():
        for pair in val:
            pair_set.add((key, pair))
    return pair_set


class TempConfig:
    def __init__(self, config):
        self.__dict__ = config

    def save(self):
        with open(self.log_path, 'w', encoding='UTF-8') as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("{}, {}\n".format(key, str(value)))


def update_default_config(conf):
    config_func = conf.conf_name
    print(conf.conf_name)
    new_config = eval(config_func + '_conf')
    print(conf)
    for key, val in new_config.items():
        conf.__dict__[key] = val