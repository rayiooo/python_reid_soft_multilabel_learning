import numpy as np
import random
import torch


class AverageMeter(object):
    '''计算存储均值和当前值'''
    
    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
def eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views=None,
                 probe_views=None, ignore_MAP=True, show_example=False):
    """
    :param dist: 2-d np array, shape=(num_gallery, num_probe), distance matrix.
    :param gallery_labels: np array, shape=(num_gallery,)
    :param probe_labels:
    :param gallery_views: np array, shape=(num_gallery,) if specified, for any probe image,
    the gallery correct matches from the same view are ignored.
    :param probe_views: must be specified if gallery_views are specified.
    :param ignore_MAP: is True, only compute cmc
    :param show_example: 是否返回示例结果
    :return:
    CMC: np array, shape=(num_probe, num_gallery). Measured by percentage
    MAP: np array, shape=(1,). Measured by percentage
    example: 示例结果labels
    """
    gallery_labels = np.asarray(gallery_labels)
    probe_labels = np.asarray(probe_labels)
    dist = np.asarray(dist)
    if show_example:
        example = []

    is_view_sensitive = False
    num_gallery = gallery_labels.shape[0]
    num_probe = probe_labels.shape[0]
    if gallery_views is not None or probe_views is not None:
        assert gallery_views is not None and probe_views is not None, \
            'gallery_views and probe_views must be specified together. \n'
        gallery_views = np.asarray(gallery_views)
        probe_views = np.asarray(probe_views)
        is_view_sensitive = True
    cmc = np.zeros((num_probe, num_gallery))
    ap = np.zeros((num_probe,))
    for i in range(num_probe):  # 对于第i号probe
        cmc_ = np.zeros((num_gallery))
        dist_ = dist[:, i]
        probe_label = probe_labels[i]
        gallery_labels_ = gallery_labels
        if is_view_sensitive:  # 同摄像头同一人不算(why?)
            probe_view = probe_views[i]
            is_from_same_view = gallery_views == probe_view
            is_correct = gallery_labels == probe_label
            should_be_excluded = is_from_same_view & is_correct
            dist_ = dist_[~should_be_excluded]
            gallery_labels_ = gallery_labels_[~should_be_excluded]
        ranking_list = np.argsort(dist_)
        inference_list = gallery_labels_[ranking_list]
        positions_correct_tuple = np.nonzero(probe_label == inference_list)
        positions_correct = positions_correct_tuple[0]
        pos_first_correct = positions_correct[0]
        cmc_[pos_first_correct:] = 1
        cmc[i] = cmc_

        if not ignore_MAP:
            num_correct = positions_correct.shape[0]  # 该label下图数
            for j in range(num_correct):
                last_precision = float(j) / float(positions_correct[j]) if j != 0 else 1.0  # 
                current_precision = float(j + 1) / float(positions_correct[j] + 1)
                ap[i] += (last_precision + current_precision) / 2.0 / float(num_correct)
                
        if show_example:
            example.append({'tgt': probe_label, 'res': inference_list[:10]})

    CMC = np.mean(cmc, axis=0)
    MAP = np.mean(ap)
    if show_example:
        example = random.sample(example, 10)
        return CMC * 100, MAP * 100, example
    return CMC * 100, MAP * 100
    

def extract_features(loader, model, index_feature=None, return_numpy=True):
    """
    extract features for the given loader using the given model
    if loader.dataset.require_view is False, the returned 'views' are empty.
    :param loader: a ReIDDataset that has attribute require_view
    :param model: returns a tuple containing the feature or only return the feature. if latter, index_feature be None
    model can also be a tuple of nn.Module, indicating that the feature extraction is multi-stage.
    in this case, index_feature should be a tuple of the same size.
    :param index_feature: in the tuple returned by model, the index of the feature.
    if the model only returns feature, this should be set to None.
    :param return_numpy: if True, return numpy array; otherwise return torch tensor
    :return: features, labels, views, np array
    """
    if type(model) is not tuple:
        models = (model,)
        indices_feature = (index_feature,)
    else:
        assert len(model) == len(index_feature)
        models = model
        indices_feature = index_feature
    for m in models:
        m.eval()

    labels = []
    views = []
    features = []

    require_view = loader.dataset.require_view
    for i, data in enumerate(loader):
        imgs = data[0].cuda()
        label_batch = data[1]
        inputs = imgs
        for m, feat_idx in zip(models, indices_feature):
            with torch.no_grad():
                output_tuple = m(inputs)
            feature_batch = output_tuple if feat_idx is None else output_tuple[feat_idx]
            inputs = feature_batch

        features.append(feature_batch)
        labels.append(label_batch)
        if require_view:
            view_batch = data[2]
            views.append(view_batch)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    views = torch.cat(views, dim=0) if require_view else views
    if return_numpy:
        return np.array(features.cpu()), np.array(labels.cpu()), np.array(views.cpu())
    else:
        return features, labels, views

    
def partition_params(module, strategy, *desired_modules):
    """
    partition params into desired part and the residual
    :param module:
    :param strategy: choices are: ['bn', 'specified'].
    'bn': desired_params = bn_params
    'specified': desired_params = all params within desired_modules
    :param desired_modules: strings, each corresponds to a specific module
    :return: two lists
    """
    if strategy == 'bn':
        desired_params_set = set()
        for m in module.modules():
            if (isinstance(m, torch.nn.BatchNorm1d) or
                    isinstance(m, torch.nn.BatchNorm2d) or
                    isinstance(m, torch.nn.BatchNorm3d)):
                desired_params_set.update(set(m.parameters()))
    elif strategy == 'specified':
        desired_params_set = set()
        for module_name in desired_modules:
            sub_module = module.__getattr__(module_name)
            for m in sub_module.modules():
                desired_params_set.update(set(m.parameters()))
    else:
        assert False, 'unknown strategy: {}'.format(strategy)
    all_params_set = set(module.parameters())
    other_params_set = all_params_set.difference(desired_params_set)
    desired_params = list(desired_params_set)
    other_params = list(other_params_set)
    return desired_params, other_params
