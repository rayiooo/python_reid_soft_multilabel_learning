import torch.nn as nn


class AgentLoss(nn.Module):
    def __init__(self):
        super(AgentLoss, self).__init__()

    def forward(self, features, agents, labels):
        pass


class DiscriminativeLoss(nn.Module):
    def __init__(self, mining_ratio=0.001):
        super(DiscriminativeLoss, self).__init__()
        pass

    def forward(self, features, multilabels, labels):
        pass


class JointLoss(nn.Module):
    def __init__(self, margin=1):
        super(JointLoss, self).__init__()
        pass

    def forward(self, features, agents, labels, similarity, features_target, similarity_target):
        pass


class MultilabelLoss(nn.Module):
    def __init__(self, batch_size, use_std=True):
        super(MultilabelLoss, self).__init__()
        pass

    def forward(self, log_multilabels, views):
        pass
