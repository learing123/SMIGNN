import torch.nn as nn
import dgl.function as fn
import torch
import torch.nn.functional as F

class SocialConv(nn.Module):
    def __init__(self):
        super(SocialConv, self).__init__()

    def forward(self, graph, user_emb):
        graph = graph.local_var()
        graph.nodes['user'].data['feat'] = user_emb
        graph.update_all(fn.copy_src('feat', 'n_feat'), fn.mean('n_feat', 'new_feat'), etype='friend')
        return graph.nodes['user'].data['new_feat']


class RatingConv(nn.Module):
    def __init__(self, conf, emb_dim):
        super(RatingConv, self).__init__()
        self.conf = conf

    def forward(self, graph, user_emb, item_emb, u_sw, i_sw):
        graph = graph.local_var()
        graph.nodes['user'].data['feat'] = user_emb
        graph.nodes['item'].data['feat'] = item_emb

        graph.update_all(fn.copy_src('feat', 'n_feat'), fn.mean('n_feat', 'new_f'), etype='like')
        graph.update_all(fn.copy_src('feat', 'n_feat'), fn.mean('n_feat', 'new_f'), etype='rev_like')

        graph.nodes['user'].data['feat'] = graph.nodes['user'].data['new_f'] + graph.nodes['user'].data['feat'] * u_sw
        graph.nodes['item'].data['feat'] = graph.nodes['item'].data['new_f'] + graph.nodes['item'].data['feat'] * i_sw

        graph.update_all(fn.copy_src('feat', 'n_feat'), fn.mean('n_feat', 'new_f'), etype='like')
        graph.update_all(fn.copy_src('feat', 'n_feat'), fn.mean('n_feat', 'new_f'), etype='rev_like')

        graph.nodes['user'].data['feat'] = graph.nodes['user'].data['new_f'] + graph.nodes['user'].data['feat'] * u_sw
        graph.nodes['item'].data['feat'] = graph.nodes['item'].data['new_f'] + graph.nodes['item'].data['feat'] * i_sw

        return graph.nodes['user'].data['feat'], graph.nodes['item'].data['feat']

class attention(nn.Module):
    def __init__(self, embedding_dim, droprate, cuda = "cpu"):
        super(attention, self).__init__()

        self.embed_dim = embedding_dim
        self.droprate = droprate
        self.device = cuda

        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax()

    def forward(self, feature1, feature2, n_neighs):
        feature2_reps = feature2.repeat(n_neighs, 1)

        x = torch.cat((feature1, feature2_reps), 1)
        x = F.relu(self.att1(x).to(self.device), inplace = True)
        x = F.dropout(x, training =self.training, p = self.droprate)
        x = self.att2(x).to(self.device)

        att = F.softmax(x, dim = 0)
        return att