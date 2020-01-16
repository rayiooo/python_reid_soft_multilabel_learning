import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist  # 一组向量相互距离处理器


class DiscriminativeLoss(nn.Module):
    '''Loss-1 MDL
    依据挖掘率和多标签，划分正负集合，拉近与拉远。
    L = -log(P / (P+N))
    P = (1 / |P|) Σ e^(-||f1 - f2||^2)
    '''
    def __init__(self, mining_ratio=0.001):
        super(DiscriminativeLoss, self).__init__()
        self.mining_ratio = mining_ratio
        self.register_buffer('n_pos_pairs', torch.Tensor([0]))
        self.register_buffer('rate_TP', torch.Tensor([0]))
        self.moment = 0.1
        self.initialized = False

    def init_threshold(self, pairwise_agreements):
        '''
        按挖掘率初始化正样本分割点
        :param pairwise_agreements: 相似比较特征
        '''
        pos = int(len(pairwise_agreements) * self.mining_ratio)
        sorted_agreements = np.sort(pairwise_agreements)
        t = sorted_agreements[-pos]
        self.register_buffer('threshold', torch.Tensor([t]).cuda())  # 正样本分割点
        self.initialized = True
        
    def forward(self, features, multilabels, labels):
        """
        :param features: shape=(BS, dim)
        :param multilabels: (BS, n_class)
        :param labels: (BS,)
        :return:
        """
        P, N = self._partition_sets(features.detach(), multilabels, labels)
        if P is None:
            pos_exponant = torch.Tensor([1]).cuda()
            num = 0
        else:
            sdist_pos_pairs = []
            for (i, j) in zip(P[0], P[1]):
                sdist_pos_pair = (features[i] - features[j]).pow(2).sum()
                sdist_pos_pairs.append(sdist_pos_pair)
            pos_exponant = torch.exp(- torch.stack(sdist_pos_pairs)).mean()
            num = -torch.log(pos_exponant)
        if N is None:
            neg_exponant = torch.Tensor([0.5]).cuda()
        else:
            sdist_neg_pairs = []
            for (i, j) in zip(N[0], N[1]):
                sdist_neg_pair = (features[i] - features[j]).pow(2).sum()
                sdist_neg_pairs.append(sdist_neg_pair)
            neg_exponant = torch.exp(- torch.stack(sdist_neg_pairs)).mean()
        den = torch.log(pos_exponant + neg_exponant)
        loss = num + den
        del pos_exponant, neg_exponant, num, den
        return loss.requires_grad_()
    
    def _partition_sets(self, features, multilabels, labels):
        '''获得batch中的正负set'''
        f = features.cpu().numpy()
        ml = multilabels.cpu().numpy()
        p_dist = pdist(f)  # f距离
        p_agree = 1 - pdist(ml, 'minkowski', p=1) / 2  # 软标签相似比较特征
        sort_idx = np.argsort(p_dist)  # 排序的idx
        n_similar = int(len(p_dist) * self.mining_ratio)  # 取前几
        similar_idx = sort_idx[:n_similar]  # 困难样本
        is_positive = p_agree[similar_idx] > self.threshold.item()  # 正样本真假表
        pos_idx = similar_idx[is_positive]
        neg_idx = similar_idx[~is_positive]
        P = dist_idx_to_pair_idx(len(f), pos_idx)
        N = dist_idx_to_pair_idx(len(f), neg_idx)
        self._update_threshold(p_agree)
        self._update_buffers(P, labels)
        return P, N
        
    def _update_threshold(self, pairwise_agreements):
        '''按一定比例更新正样本分割点'''
        pos = int(len(pairwise_agreements) * self.mining_ratio)
        sorted_agreements = np.sort(pairwise_agreements)
        t = torch.Tensor([sorted_agreements[-pos]]).cuda()
        self.threshold = self.threshold * (1 - self.moment) + t * self.moment
        del t

    def _update_buffers(self, P, labels):  # ？labels is what
        if P is None:
            self.n_pos_pairs = 0.9 * self.n_pos_pairs
            return 0
        n_pos_pairs = len(P[0])
        count = 0
        for (i, j) in zip(P[0], P[1]):
            count += labels[i] == labels[j]  # 正对
        rate_TP = float(count) / n_pos_pairs
        self.n_pos_pairs = 0.9 * self.n_pos_pairs + 0.1 * n_pos_pairs
        self.rate_TP = 0.9 * self.rate_TP + 0.1 * rate_TP


class MultilabelLoss(nn.Module):
    '''Loss-2 CML'''
    def __init__(self, batch_size, use_std=True):
        super(MultilabelLoss, self).__init__()
        self.use_std = use_std
        self.moment = batch_size / 10000
        self.initialized = False
        
    def init_centers(self, log_multilabels, views):
        ml_mean = []
        ml_std = []
        for v in torch.unique(views):
            mls_in_view = log_multilabels[views == v]
            if len(mls_in_view) == 1:
                continue
            mean = mls_in_view.mean(dim=0)
            std = mls_in_view.std(dim=0)
            ml_mean.append(mean)
            ml_std.append(std)
        center_mean = torch.mean(torch.stack(ml_mean), dim=0)
        center_std = torch.mean(torch.stack(ml_std), dim=0)
        self.register_buffer('center_mean', center_mean)
        self.register_buffer('center_std', center_std)
        self.initialized = True
    
    def _update_centers(self, log_multilabels, views):
        '''
        更新中心分布.
        :param log_multilabels: 不要记录该参数的梯度.
        '''
        means = []
        stds = []
        for v in torch.unique(views):
            mls_in_view = log_multilabels[views == v]
            if len(mls_in_view) == 1:
                continue
            mean = mls_in_view.mean(dim=0)
            means.append(mean)
            if self.use_std:
                std = mls_in_view.std(dim=0)
                stds.append(std)
        new_mean = torch.mean(torch.stack(means), dim=0)
        self.center_mean = self.center_mean * (1 - self.moment) + new_mean * self.moment
        if self.use_std:
            new_std = torch.mean(torch.stack(stds), dim=0)
            self.center_std = self.center_std * (1 - self.moment) + new_std * self.moment

    def forward(self, log_multilabels, views):
        self._update_centers(log_multilabels.detach(), views)
        
        loss_terms = []
        for v in torch.unique(views):
            mls_in_view = log_multilabels[views == v]
            if len(mls_in_view) == 1:
                continue
            mean = mls_in_view.mean(dim=0)
            loss_mean = (mean - self.center_mean).pow(2).sum()
            loss_terms.append(loss_mean)
            if self.use_std:
                std = mls_in_view.std(dim=0)
                loss_std = (std - self.center_std).pow(2).sum()
                loss_terms.append(loss_std)
        loss_total = torch.mean(torch.stack(loss_terms))
        return loss_total


class JointLoss(nn.Module):
    '''Loss-3-2 RJ'''
    def __init__(self, margin=1):
        super(JointLoss, self).__init__()
        self.margin = margin
        self.sim_margin = 1 - margin / 2

    def forward(self, agents, a_f, a_sim, ay, b_f, b_sim):
        loss_terms = []
        arange = torch.arange(len(agents)).cuda()
        zero = torch.Tensor([0]).cuda()
        # 有监督数据集
        for (f, y, s) in zip(a_f, ay, a_sim):
            # 正样本 f & agent 拉近
            loss_pos = (f - agents[y]).pow(2).sum()
            loss_terms.append(loss_pos)
            # 难负样本 f & agent 拉远
            hard_agent_idx = (arange != y) & (s > self.sim_margin)  # 非此人但软标签相似(难负样本)
            if torch.any(hard_agent_idx):  # 如果有 True
                hard_neg_sdist = (f - agents[hard_agent_idx]).pow(2).sum(dim=1)
                loss_neg = torch.max(zero, self.margin - hard_neg_sdist).mean()
                loss_terms.append(loss_neg)
        # 无监督与有监督数据集中困难负样本拉远
        for (f, s) in zip(b_f, b_sim):
            hard_agent_idx = s > self.sim_margin
            if torch.any(hard_agent_idx):
                hard_neg_sdist = (f - agents[hard_agent_idx]).pow(2).sum(dim=1)
                loss_neg = torch.max(zero, self.margin - hard_neg_sdist).mean()
                loss_terms.append(loss_neg)
        loss_total = torch.mean(torch.stack(loss_terms))
        return loss_total

    
def pair_idx_to_dist_idx(d, i, j):
    """
    :param d: numer of elements
    :param i: np.array. i < j in every element
    :param j: np.array
    :return:
    """
    assert np.sum(i < j) == len(i)
    index = d * i - i * (i + 1) / 2 + j - 1 - i
    return index.astype(int)


def dist_idx_to_pair_idx(d, i):
    """
    从相互距离中恢复出每一对特征的idx。
    :param d: number of samples
    :param i: np.array
    :return: x_nparr, y_nparr
    """
    if i.size == 0:
        return None
    b = 1 - 2 * d
    x = np.floor((-b - np.sqrt(b ** 2 - 8 * i)) / 2).astype(int)
    y = (i + x * (b + x + 2) / 2 + 1).astype(int)
    return x, y
