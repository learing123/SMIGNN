import torch
import torch.nn as nn
from model.layers import SocialConv, RatingConv
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = torch.tanh(self.linear(x))
        alpha = torch.softmax(self.attention(h), dim=0)
        attended = torch.sum(alpha * x, dim=0)
        return attended

class UserRating(nn.Module):
    def __init__(self, conf, data):
        super(UserRating, self).__init__()
        self.conf = conf
        self.data = data
        self.user_emb = nn.Parameter(torch.normal(mean=0, std=0.01, size=(conf.num_users, conf.emb_dim)))
        self.item_emb = nn.Parameter(torch.normal(mean=0, std=0.01, size=(conf.num_items, conf.emb_dim)))

        self.rating_layer = RatingConv(conf, conf.emb_dim)

    def forward(self, users, items, users_list=None, items_list=None, mode='test'):
        graph = self.data.data_graph.local_var()
        if mode == 'train':
            remove_iu_eid_list = []
            remove_ui_eid_list = []
            for idx in range(len(users_list)):
                iu_eid = self.data.train_iu_pair2eid.get((items_list[idx], users_list[idx]), -1)
                ui_eid = self.data.train_ui_pair2eid.get((users_list[idx], items_list[idx]), -1)
                if iu_eid >= 0:
                    remove_iu_eid_list.append(iu_eid)
                if ui_eid >= 0:
                    remove_ui_eid_list.append(ui_eid)
            graph.remove_edges(remove_iu_eid_list, 'rev_like')
            graph.remove_edges(remove_ui_eid_list, 'like')

        user_deg = graph.out_degrees(etype='like').float().to(self.conf.device).unsqueeze(1)
        u_sw = 1 - user_deg / (user_deg + 1e-8)
        item_deg = graph.out_degrees(etype='rev_like').float().to(self.conf.device).unsqueeze(1)
        i_sw = 1 - item_deg / (item_deg + 1e-8)

        fuse_user_emb = self.user_emb
        fuse_item_emb = self.item_emb

        user_emb_gnn1, item_emb_gnn1 = self.rating_layer(graph, fuse_user_emb, fuse_item_emb, u_sw, i_sw)
        user_emb_gnn2, item_emb_gnn2 = self.rating_layer(graph, user_emb_gnn1, item_emb_gnn1, u_sw, i_sw)

        final_user_emb = 0
        if 0 in self.conf.l_user:
            final_user_emb = final_user_emb + fuse_user_emb
        if 1 in self.conf.l_user:
            final_user_emb = final_user_emb + user_emb_gnn1
        if 2 in self.conf.l_user:
            final_user_emb = final_user_emb + user_emb_gnn2

        final_item_emb = 0
        if 0 in self.conf.l_item:
            final_item_emb = final_item_emb + fuse_item_emb
        if 1 in self.conf.l_item:
            final_item_emb = final_item_emb + item_emb_gnn1
        if 2 in self.conf.l_item:
            final_item_emb = final_item_emb + item_emb_gnn2

        latest_user_emb = final_user_emb[users]
        latest_item_emb = final_item_emb[items]

        predict = torch.sigmoid(torch.sum(torch.mul(latest_user_emb, latest_item_emb), dim=1))

        return predict, latest_user_emb, latest_item_emb

class UserSocial(nn.Module):
    def __init__(self, conf, data):
        super(UserSocial, self).__init__()
        self.conf = conf
        self.data = data
        self.user_emb = nn.Parameter(torch.normal(mean=0, std=0.01, size=(conf.num_users, conf.emb_dim)))
        self.item_emb = nn.Parameter(torch.normal(mean=0, std=0.01, size=(conf.num_items, conf.emb_dim)))

        self.social_layer = SocialConv()

    def forward(self, users, items, users_list=None, items_list=None, mode='test'):
        graph = self.data.data_graph.local_var()
        if mode == 'train':
            remove_iu_eid_list = []
            remove_ui_eid_list = []
            for idx in range(len(users_list)):
                iu_eid = self.data.train_iu_pair2eid.get((items_list[idx], users_list[idx]), -1)
                ui_eid = self.data.train_ui_pair2eid.get((users_list[idx], items_list[idx]), -1)
                if iu_eid >= 0:
                    remove_iu_eid_list.append(iu_eid)
                if ui_eid >= 0:
                    remove_ui_eid_list.append(ui_eid)
            graph.remove_edges(remove_iu_eid_list, 'rev_like')
            graph.remove_edges(remove_ui_eid_list, 'like')

        fuse_user_emb = self.user_emb
        fuse_item_emb = self.item_emb

        user_emb_gnn1 = self.social_layer(graph, fuse_user_emb)
        user_emb_gnn2 = self.social_layer(graph, user_emb_gnn1)

        final_user_emb = 0
        if 0 in self.conf.l_user:
            final_user_emb = final_user_emb + fuse_user_emb
        if 1 in self.conf.l_user:
            final_user_emb = final_user_emb + user_emb_gnn1
        if 2 in self.conf.l_user:
            final_user_emb = final_user_emb + user_emb_gnn2
        final_item_emb = fuse_item_emb

        latest_user_emb = final_user_emb[users]
        latest_item_emb = final_item_emb[items]

        predict = torch.sigmoid(torch.sum(torch.mul(latest_user_emb, latest_item_emb), dim=1))

        return predict, latest_user_emb, latest_item_emb

class UserFusion(nn.Module):
    def __init__(self, conf, data):
        super(UserFusion, self).__init__()
        self.conf = conf
        self.data = data
        self.user_emb = nn.Parameter(torch.normal(mean=0, std=0.01, size=(conf.num_users, conf.emb_dim)))
        self.item_emb = nn.Parameter(torch.normal(mean=0, std=0.01, size=(conf.num_items, conf.emb_dim)))

        self.social_layer = SocialConv()
        self.rating_layer = RatingConv(conf, conf.emb_dim)
        self.attention = AttentionLayer(conf.emb_dim, conf.emb_dim)

    def forward(self, users, items, users_list=None, items_list=None, mode='test'):
        graph = self.data.data_graph.local_var()
        if mode == 'train':
            remove_iu_eid_list = []
            remove_ui_eid_list = []
            for idx in range(len(users_list)):
                iu_eid = self.data.train_iu_pair2eid.get((items_list[idx], users_list[idx]), -1)
                ui_eid = self.data.train_ui_pair2eid.get((users_list[idx], items_list[idx]), -1)
                if iu_eid >= 0:
                    remove_iu_eid_list.append(iu_eid)
                if ui_eid >= 0:
                    remove_ui_eid_list.append(ui_eid)
            graph.remove_edges(remove_iu_eid_list, 'rev_like')
            graph.remove_edges(remove_ui_eid_list, 'like')

        fuse_user_emb = self.user_emb
        fuse_item_emb = self.item_emb

        user_deg = graph.out_degrees(etype='like').float().to(self.conf.device).unsqueeze(1)
        u_sw = 1 - user_deg / (user_deg + 1e-8)
        item_deg = graph.out_degrees(etype='rev_like').float().to(self.conf.device).unsqueeze(1)
        i_sw = 1 - item_deg / (item_deg + 1e-8)

        social_user_emb_gnn1 = self.social_layer(graph, fuse_user_emb)
        rating_user_emb_gnn1, rating_item_emb_gnn1 = self.rating_layer(graph, fuse_user_emb, fuse_item_emb, u_sw, i_sw)

        user_emb_gnn1 = 0.5 * social_user_emb_gnn1 + 0.5 * rating_user_emb_gnn1

        social_user_emb_gnn2 = self.social_layer(graph, social_user_emb_gnn1)
        rating_user_emb_gnn2, light_item_emb_gnn2 = self.rating_layer(graph, rating_user_emb_gnn1, rating_item_emb_gnn1,
                                                                      u_sw, i_sw)
        user_emb_gnn2 = 0.5 * social_user_emb_gnn2 + 0.5 * rating_user_emb_gnn2

        final_user_emb = 0
        if 0 in self.conf.l_user:
            final_user_emb = final_user_emb + fuse_user_emb
        if 1 in self.conf.l_user:
            final_user_emb = final_user_emb + user_emb_gnn1
        if 2 in self.conf.l_user:
            final_user_emb = final_user_emb + user_emb_gnn2

        final_item_emb = fuse_item_emb
        latest_user_emb = final_user_emb[users]
        latest_item_emb = final_item_emb[items]

        predict = torch.sigmoid(torch.sum(torch.mul(latest_user_emb, latest_item_emb), dim=1))

        return predict, latest_user_emb, latest_item_emb
