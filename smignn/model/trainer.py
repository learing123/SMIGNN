import time
import os
import numpy as np
from model.SoRaGAT import UserRating
from model.SoRaGAT import UserSocial
from model.SoRaGAT import UserFusion
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Process, Queue
import math
import torch

class Trainer:
    def __init__(self, conf, data):
        self.conf = conf
        self.data = data
        self.model_dict = self.get_model_dict(conf.model_list)
        self.evaluate = Evaluate(conf.top_k)

    @staticmethod
    def get_model(conf, data, model_name):
        if model_name == 'fusion':
            return UserFusion(conf, data)
        elif model_name == 'social':
            return UserSocial(conf, data)
        elif model_name == 'rating':
            return UserRating(conf, data)
        else:
            raise NotImplementedError('Invalid model', conf.model)

    def get_model_dict(self, model_list):
        model_dict = {}
        for model_name in model_list:
            model = self.get_model(self.conf, self.data, model_name).to(self.conf.device)
            model_dict[model_name] = {
                'model': model,
                'model_path': os.path.join(self.conf.cur_out_path, model_name + '.pkt'),
                'optimizer': torch.optim.Adam(model.parameters(), lr=self.conf.lr),
                'criterion': torch.nn.BCELoss(),
                'best_perform': {
                    'train_hit': 0.0,
                    'train_ndcg': 0.0,
                    'valid_hit': 0.0,
                    'valid_ndcg': 0.0,
                    'test_hit': 0.0,
                    'test_ndcg': 0.0,
                },
                'cur_perform': {
                    'train_hit': 0.0,
                    'train_ndcg': 0.0,
                    'valid_hit': 0.0,
                    'valid_ndcg': 0.0,
                    'test_hit': 0.0,
                    'test_ndcg': 0.0,
                }
            }
        return model_dict

    def train_epoch(self, data, model_dict):
        data_loader = data.train_data_loader
        for step, batch_data in enumerate(tqdm(data_loader, desc="Iteration")):
            users_list, items_list, labels = batch_data
            users = torch.tensor(users_list).to(self.conf.device).long()
            items = torch.tensor(items_list).to(self.conf.device).long()
            labels = torch.tensor(labels).to(self.conf.device).float()

            batch_loss = 0
            pre_dict = {}
            for model in model_dict.keys():
                predict, user_emb, item_emb = \
                    model_dict[model]['model'](users, items, users_list, items_list, mode='train')
                model_loss = model_dict[model]['criterion'](predict, labels)
                batch_loss = batch_loss + model_loss
                pre_dict[model] = {'pre': predict, 'user_emb': user_emb, 'item_emb': item_emb}

            for (f_model, s_model), distill_weight in self.conf.distill_dict.items():
                f_pre = pre_dict[f_model]['pre']
                s_pre = pre_dict[s_model]['pre']
                distill_loss = compute_pre_distill_loss(f_pre, s_pre) * distill_weight
                batch_loss = batch_loss + distill_loss

            for model in model_dict.keys():
                model_dict[model]['optimizer'].zero_grad()
            batch_loss.backward()
            for model in model_dict.keys():
                model_dict[model]['optimizer'].step()

    @staticmethod
    def get_eval_metrics_single_process(message_q, user_list, positive_predict_dict, negative_predict_dict,
                                        evaluate, top_k):
        hit_k_list = []
        ndcg_k_list = []
        for user in user_list:
            hit_k, ndcg_k = evaluate.get_hit_ndcg(positive_predict_dict[user], negative_predict_dict[user], top_k)
            hit_k_list.append(hit_k)
            ndcg_k_list.append(ndcg_k)
        mean_hit_k = np.mean(hit_k_list)
        mean_ndcg_k = np.mean(ndcg_k_list)
        message_q.put((mean_hit_k, mean_ndcg_k, len(hit_k_list)))

    def get_eval_metrics_multi_process(self, user_list, positive_predict_dict, negative_predict_dict):
        message_q = Queue()

        batch_size = len(user_list) // self.conf.num_proc + 1
        index = 0
        process_list = []
        for _ in range(self.conf.num_proc):
            if index + batch_size < len(user_list):
                batch_user_list = user_list[index:index + batch_size]
                index = index + batch_size
            else:
                batch_user_list = user_list[index:len(user_list)]
            p = Process(target=self.get_eval_metrics_single_process,
                        args=(message_q, batch_user_list, positive_predict_dict, negative_predict_dict, self.evaluate, self.conf.top_k))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()

        hit_k_sum = 0.0
        ndcg_k_sum = 0.0
        num_user_sum = 0.0
        for _ in range(self.conf.num_proc):
            mean_hit_k, mean_ndcg_k, num_user = message_q.get()
            hit_k_sum += mean_hit_k * num_user
            ndcg_k_sum += mean_ndcg_k * num_user
            num_user_sum += num_user
        mean_hit_k = hit_k_sum / num_user_sum
        mean_ndcg_k = ndcg_k_sum / num_user_sum

        return mean_hit_k, mean_ndcg_k

    def eval_net(self, net, user_idx_dict, user_list, item_list, neg_data_loader, criterion, mode='train'):
        net.eval()
        if mode == 'train':
            num_negatives = self.conf.num_eval_negatives
        else:
            num_negatives = self.conf.num_test_negatives

        with torch.no_grad():
            positive_users = torch.tensor(user_list).to(self.conf.device).long()
            positive_items = torch.tensor(item_list).to(self.conf.device).long()
            positive_labels = torch.ones_like(positive_users).to(self.conf.device).float()
            positive_predicts, _, _ = net(positive_users, positive_items, mode='eval')
            positive_loss = criterion(positive_predicts, positive_labels).item()
            positive_predicts = positive_predicts.cpu().numpy()

            positive_predict_dict = defaultdict(list)
            negative_predict_dict = defaultdict(list)

            for user_id in user_list:
                positive_predict_dict[user_id] = positive_predicts[user_idx_dict[user_id]]

            negative_loss = 0.0
            num_negative_users = 0
            for step, data in enumerate(tqdm(neg_data_loader, desc="Iteration")):
                users_idx_list, negative_users, negative_items = data
                negative_users = torch.tensor(negative_users).to(self.conf.device).long()
                negative_items = torch.tensor(negative_items).to(self.conf.device).long()
                negative_labels = torch.zeros_like(negative_users).to(self.conf.device).float()
                negative_predicts, _, _ = net(negative_users, negative_items, mode='eval')
                negative_loss += criterion(negative_predicts, negative_labels).item() * len(negative_users)
                num_negative_users += len(negative_users)
                negative_predicts = negative_predicts.cpu().numpy().reshape(-1, num_negatives)

                for idx, user_id in enumerate(users_idx_list):
                    negative_predict_dict[user_id] = negative_predicts[idx]
            negative_loss = negative_loss / num_negative_users

            t1 = time.time()
            mean_hit_k, mean_ndcg_k = self.get_eval_metrics_multi_process(
                user_list, positive_predict_dict, negative_predict_dict
            )
            t2 = time.time()
            print('Eval_time', t2 - t1)

        return positive_loss, negative_loss, mean_hit_k, mean_ndcg_k

    def eval_epoch(self, epoch, data, model_dict, save_model=False):
        with open(self.conf.log_path, 'a') as f:
            f.write('epoch: {}\n'.format(epoch))
        for model in model_dict.keys():
            train_pos_loss, train_neg_loss, train_hit, train_ndcg = self.eval_net(
                model_dict[model]['model'],
                data.train_eval_user_idx_dict,
                data.train_eval_user_list,
                data.train_eval_item_list,
                data.train_neg_data_loader_test,
                model_dict[model]['criterion'],
                mode='test',
            )
            valid_pos_loss, valid_neg_loss, valid_hit, valid_ndcg = self.eval_net(
                model_dict[model]['model'],
                data.valid_eval_user_idx_dict,
                data.valid_eval_user_list,
                data.valid_eval_item_list,
                data.valid_neg_data_loader_test,
                model_dict[model]['criterion'],
                mode='train')
            test_pos_loss, test_neg_loss, test_hit, test_ndcg = self.eval_net(
                model_dict[model]['model'],
                self.data.test_eval_user_idx_dict,
                self.data.test_eval_user_list,
                self.data.test_eval_item_list,
                self.data.test_neg_data_loader_test,
                model_dict[model]['criterion'],
                mode='train')

            model_dict[model]['cur_perform'] = {
                'train_hit': train_hit,
                'train_ndcg': test_ndcg,
                'valid_hit': valid_hit,
                'valid_ndcg':valid_ndcg,
                'test_hit': test_hit,
                'test_ndcg': test_ndcg
            }
            if model_dict[model]['cur_perform']['valid_ndcg'] >= model_dict[model]['best_perform']['valid_ndcg']:
                model_dict[model]['best_perform'] = {
                    'train_hit': train_hit,
                    'train_ndcg': test_ndcg,
                    'valid_hit': valid_hit,
                    'valid_ndcg':valid_ndcg,
                    'test_hit': test_hit,
                    'test_ndcg': test_ndcg
                }
                if save_model:
                    torch.save(model_dict[model]['model'].state_dict(), model_dict[model]['model_path'])
            print(('Model: {}, '
                        'train_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}, '
                        'valid_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}, '
                        'test_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}\n'
                        .format(model, train_pos_loss, train_neg_loss, train_hit, train_ndcg,
                                valid_pos_loss, valid_neg_loss, valid_hit, valid_ndcg,
                                test_pos_loss, test_neg_loss, test_hit, test_ndcg)))

            with open(self.conf.log_path, 'a') as f:
                f.write('Model: {}, '
                        'train_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}, '
                        'valid_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}, '
                        'test_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}\n'
                        .format(model, train_pos_loss, train_neg_loss, train_hit, train_ndcg,
                                valid_pos_loss, valid_neg_loss, valid_hit, valid_ndcg,
                                test_pos_loss, test_neg_loss, test_hit, test_ndcg))

    def train(self):
        for epoch in range(1, self.conf.epochs + 1):
            print('epoch: ', epoch)
            self.train_epoch(self.data, self.model_dict)
            if epoch % self.conf.eval_epochs == 0:
                self.eval_epoch(epoch, self.data, self.model_dict, save_model=True)

    def test(self):
        with open(self.conf.result_path, 'w') as f:
            f.write('Test_result\n')
        for model in self.model_dict.keys():
            self.model_dict[model]['model'].load_state_dict(torch.load(self.model_dict[model]['model_path']))
            test_pos_loss, test_neg_loss, test_hit, test_ndcg = self.eval_net(
                self.model_dict[model]['model'],
                self.data.test_eval_user_idx_dict,
                self.data.test_eval_user_list,
                self.data.test_eval_item_list,
                self.data.test_neg_data_loader_test,
                self.model_dict[model]['criterion'],
                mode='test')
            print('Model: {} test set - average loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}'
                  .format(model, test_pos_loss, test_neg_loss, test_hit, test_ndcg))
            with open(self.conf.result_path, 'a') as f:
                f.write('Model: {} test set - average loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}\n'
                        .format(model, test_pos_loss, test_neg_loss, test_hit, test_ndcg))

class Evaluate:
    def __init__(self, max_length):
        self.idcg_list = self.get_idcg_list(max_length)

    @staticmethod
    def get_idcg_list(max_length):
        idcg_list = [0]
        idcg = 0.0
        for idx in range(max_length):
            idcg = idcg + math.log(2) / math.log(idx + 2)
            idcg_list.append(idcg)
        return idcg_list

def compute_pre_distill_loss(pre_a, pre_b):
    distill_loss = - torch.mean(pre_b.detach() * torch.log(pre_a) + (1 - pre_b.detach()) * torch.log(1 - pre_a))
    return distill_loss
