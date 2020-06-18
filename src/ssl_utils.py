import networkx as nx
import torch
import numpy as np
import os
import subprocess
import scipy.sparse as sp

def encode_onehot(labels):
    eye = np.eye(labels.max() + 1)
    onehot_mx = eye[labels]
    return onehot_mx

def normalize_vector(v):
    # mean_ = torch.mean(v)
    # std = torch.std(v)
    # return (v - mean_) / std
    # return (v-v.min())/(v.max()-v.min())
    return v

def calc_pagerank(G, device, multi_class):
    pagerank = nx.pagerank(G, alpha=0.85)
    if multi_class:
        pseudo_labels = torch.LongTensor([p for p in pagerank.values()]).to(device)
        pseudo_labels = normalize_vector(pseudo_labels)
        sorted_values = sorted(pagerank.values(), reverse=True)
        return split_multiclass(pseudo_labels, sorted_values)
    else:
        pseudo_labels = torch.Tensor([p for p in pagerank.values()]).to(device)
        pseudo_labels = normalize_vector(pseudo_labels)
        return pseudo_labels

def calc_degree(G, device, multi_class):
    degrees = dict(G.degree())
    if multi_class:
        pseudo_labels = torch.LongTensor([p for p in degrees.values()]).to(device)
        pseudo_labels = normalize_vector(pseudo_labels)
        sorted_values = sorted(degrees.values(), reverse=True)
        return split_multiclass(pseudo_labels, sorted_values)
    else:
        pseudo_labels = torch.Tensor([p for p in degrees.values()]).to(device)
        pseudo_labels = normalize_vector(pseudo_labels)
        return pseudo_labels


def calc_node_importance(G):
    pass

def calc_centrality(G, device, multi_class):
    centrality = nx.closeness_centrality(G)
    if multi_class:
        pseudo_labels = torch.LongTensor([p for p in centrality.values()]).to(device)
        sorted_values = sorted(centrality.values(), reverse=True)
        return split_multiclass(pseudo_labels, sorted_values)
    else:
        pseudo_labels = torch.Tensor([p for p in centrality.values()]).to(device)
        return pseudo_labels

def calc_clustering_coeff(G):
    pass

def split_multiclass(pseudo_labels, sorted_values):
    n = len(sorted_values)
    # high = sorted_values[int(0.3*n)]
    # low = sorted_values[int(0.7*n)]

    high = 7
    low = 2
    high_indices = (pseudo_labels > high)
    low_indices = (pseudo_labels <= low)
    mid_indices = ((pseudo_labels > low) & (pseudo_labels <= high))
    real_mid_indices = ((pseudo_labels > sorted_values[int(0.35*n)]) & \
                        (pseudo_labels <= sorted_values[int(0.65*n)]))
    pseudo_labels[high_indices] = 2
    pseudo_labels[low_indices] = 0
    pseudo_labels[mid_indices] = 1
    pseudo_labels[real_mid_indices] = 1

    return pseudo_labels

def bfs(adj, node, visited):
    pass


def label_statics(labels):
    counter = {}
    for x in labels:
        counter[x] = counter.get(x, 0) + 1/len(labels)
    return counter

import pandas as pd
def feature_statics(features):
    df = pd.DataFrame(features)
    import ipdb
    ipdb.set_trace()


def othertraining(adj, features, labels, idx_train, args, model='LP'):
    if model =='ICA':
        from distance import ICAAgent
        unlabeled = np.array([x for x in range(adj.shape[0]) if x not in idx_train])
        labels = encode_onehot(labels)
        agent = ICAAgent(adj, features, labels, idx_train, unlabeled, args)
        preds = agent.concated

    if model == 'LP':
        from distance import LPAgent
        unlabeled = np.array([x for x in range(adj.shape[0]) if x not in idx_train])
        labels = encode_onehot(labels)
        agent = LPAgent(adj, features, labels, idx_train, unlabeled)
        preds = agent.concated
        probs = agent.probs
        nclass = labels.shape[1]

    return torch.LongTensor(preds), np.arange(adj.shape[0])[probs.max(1) > 4/nclass]


def selftraining(adj, labels, idx_train, args):
    try:
        with open('script/ssl/warmup.sh', 'r') as f:
            lines = f.readlines()
        with open('script/ssl/warmup.sh', 'w') as f:
            for l in lines:
                if 'seed' in l:
                    l = '   --seed %s\\\n' % args.seed
                if 'dataset' in l:
                    l = '   --dataset %s\\\n' % args.dataset
                f.write(l)
        subprocess.check_output('sh script/ssl/warmup.sh'.split())
    except subprocess.CalledProcessError as e:
        # print(e.output)
        assert False, 'Subprocess Call Error'

    # prediction = np.load(f'preds/{args.dataset}_{args.seed}_pred.npy')
    prediction = np.load(f'ICA_probs_{args.dataset}_{args.seed}.npy')

    labels = encode_onehot(labels)
    y_train = get_y(idx_train, labels)
    train_mask = get_mask(idx_train, labels)

    # eta = adj.shape[0]/(adj.sum()/adj.shape[0])**2
    # t = (encode_onehot(labels[idx_train]).sum(0)*3*eta/len(idx_train)).astype(np.int64)
    eta = adj.shape[0]/(adj.sum()/adj.shape[0])**2
    t = (y_train.sum(axis=0)*3*eta/y_train.sum()).astype(np.int64)

    new_gcn_index = np.argmax(prediction, axis=1)
    confidence = np.max(prediction, axis=1)
    sorted_index = np.argsort(-confidence)

    no_class = y_train.shape[1]  # number of class:
    if hasattr(t, '__getitem__'):
        assert len(t) >= no_class
        index = []
        count = [0 for i in range(no_class)]
        for i in sorted_index:
            for j in range(no_class):
                if new_gcn_index[i] == j and count[j] < t[j] and not train_mask[i]:
                    index.append(i)
                    count[j] += 1
    else:
        index = sorted_index[:t]

    indicator = np.zeros(train_mask.shape, dtype=np.bool)
    indicator[index] = True
    indicator = np.logical_and(np.logical_not(train_mask), indicator)

    prediction = np.zeros(prediction.shape)
    prediction[np.arange(len(new_gcn_index)), new_gcn_index] = 1.0
    prediction[train_mask] = y_train[train_mask]

    # correct_labels = np.sum(prediction[indicator] * all_labels[indicator], axis=0)
    correct_labels = np.sum(prediction[indicator] * labels[indicator], axis=0)
    count = np.sum(prediction[indicator], axis=0)
    print('Additiona Label:')
    for i, j in zip(correct_labels, count):
        print(int(i), '/', int(j), sep='', end='\t')
    print()

    y_train = np.copy(y_train)
    train_mask = np.copy(train_mask)
    train_mask[indicator] = 1
    y_train[indicator] = prediction[indicator]

    # return y_train, train_mask
    return torch.LongTensor(y_train.argmax(1)),\
           np.arange(adj.shape[0])[train_mask]

def get_mask(idx, labels):
    mask = np.zeros(labels.shape[0], dtype=np.bool)
    mask[idx] = 1
    return mask

def get_y(idx, labels):
    mx = np.zeros(labels.shape)
    mx[idx] = labels[idx]
    return mx


class SMP:

    def __init__(self):
        pass


from sklearn.metrics.pairwise import cosine_similarity
def label_correction(embeddings, labels, idx_sampled, idx_train=None, is_numpy=False):
    get_sim = cosine_similarity
    embeddings = embeddings.cpu().numpy()
    nclass = labels.max() + 1
    all_nodes = np.arange(len(labels))
    n_rho = 8
    corrected_labels = np.zeros((len(idx_sampled), nclass))
    for c in range(nclass):
        # sampling
        n_samples = 320
        perm = np.random.permutation(all_nodes[labels == c])
        if len(perm) < n_samples:
            n_samples = len(perm)
        else:
            perm = perm[: n_samples]
        # perm = np.random.permutation(all_nodes[labels == c])
        # n_samples = len(perm)

        # cos sim
        S = get_sim(embeddings[perm])
        # density
        S_c = np.percentile(S, 40)
        # for i in range(n_samples):
        #     rho = np.sign(S[i] - S_c).sum()
        rho = np.array([np.sign(S[i] - S_c).sum() for i in range(n_samples)])
        rho_max = rho.max()
        eta = np.empty_like(rho)

        for i in range(n_samples):
            if rho[i] < rho_max:
                eta[i] = max([S[i][j] for j in range(n_samples) if rho[j] > rho[i]])
            else:
                eta[i] = min(S[i])
        # get prototype
        rho[eta>=0.95] = 0
        prototypes = perm[np.argsort(rho)[-n_rho: ]]
        # correct label
        S_proto = get_sim(embeddings[idx_sampled], embeddings[prototypes])
        # for i in range(len(labels)):
        #     corrected_label[i][c] = S_proto[i].mean()
        corrected_labels[:, c] = S_proto.mean(1)
    if is_numpy:
        corrected_labels = corrected_labels.argmax(1)
        if idx_train is not None:
            idx_train = idx_train.cpu().numpy()
            corrected_labels[idx_train] = labels[idx_train]
    else:
        corrected_labels = torch.LongTensor(corrected_labels.argmax(1)).cuda()
        if idx_train is not None:
            labels = torch.LongTensor(labels).cuda()
            corrected_labels[idx_train] = labels[idx_train]
    return corrected_labels


def to_scipy(tensor):
    """Convert a dense/sparse tensor to """
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)

def is_sparse_tensor(tensor):
    # if hasattr(tensor, 'nnz'):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False


def get_few_labeled_splits(labels, train_size, seed=None):
    '''
        This setting follows gcn, where we randomly sample 20 instances for each class
        as training data, 500 instances as validation data, 1000 instances as test data.
    '''
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(len(labels))
    nclass = labels.max() + 1
    idx_train = []
    idx_unlabeled = []
    for i in range(nclass):
        labels_i = idx[labels==i]
        labels_i = np.random.permutation(labels_i)
        idx_train = np.hstack((idx_train, labels_i[: train_size])).astype(np.int)
        idx_unlabeled = np.hstack((idx_unlabeled, labels_i[train_size: ])).astype(np.int)

    idx_unlabeled = np.random.permutation(idx_unlabeled)
    idx_val = idx_unlabeled[: 500]
    idx_test = idx_unlabeled[500: 1500]
    return idx_train, idx_val, idx_test

def get_splits_each_class(labels, train_size, seed=None):
    '''
        This setting follows gcn, where we randomly sample n instances for class,
        where n=train_size
    '''
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(len(labels))
    nclass = labels.max() + 1
    idx_train = []
    idx_val = []
    idx_test = []
    for i in range(nclass):
        labels_i = idx[labels==i]
        labels_i = np.random.permutation(labels_i)
        idx_train = np.hstack((idx_train, labels_i[: train_size])).astype(np.int)
        idx_val = np.hstack((idx_val, labels_i[train_size: 2*train_size])).astype(np.int)
        idx_test = np.hstack((idx_test, labels_i[2*train_size: ])).astype(np.int)

    return np.random.permutation(idx_train), np.random.permutation(idx_val), \
           np.random.permutation(idx_test)


