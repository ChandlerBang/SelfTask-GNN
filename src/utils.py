import pickle as pkl
import sys
import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from normalization import fetch_normalization, row_normalize

datadir = "data"

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


def load_nell(dataset="nell.0.001", normalization="AugNormAdj", porting_to_torch=True,data_path=datadir, task_type="full"):

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/nell_data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/nell_data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'nell.0.001':
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-allx.shape[0], :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)
        if not os.path.isfile("data/{}.features.npz".format(dataset)):
            print("Creating feature vectors for relations - this might take a while...")
            features_extended = sp.hstack((features, sp.lil_matrix((features.shape[0], len(isolated_node_idx)))),
                                          dtype=np.int32).todense()

            features_extended[isolated_node_idx, features.shape[1]:] = np.eye(len(isolated_node_idx))
            features = sp.csr_matrix(features_extended)
            print("Done!")
            save_sparse_csr("data/{}.features".format(dataset), features)
        else:
            features = load_sparse_csr("data/{}.features.npz".format(dataset))

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # degree = np.asarray(G.degree)
    degree = np.sum(adj, axis=1)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if task_type == "full":
        print("Load full supervised task.")
        #supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(ally)- 500)
        idx_val = range(len(ally) - 500, len(ally))
    elif task_type == "semi":
        print("Load semi-supervised task.")
        #semi-supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)
    else:
        raise ValueError("Task type: %s is not supported. Available option: full and semi.")

    features = features.astype(float)
    adj, features = preprocess_citation(adj, features, normalization)
    features = np.array(features.todense())
    labels = np.argmax(labels, axis=1)
    # porting to pytorch
    if porting_to_torch:
        features = torch.FloatTensor(features).float()
        labels = torch.LongTensor(labels)
        # labels = torch.max(labels, dim=1)[1]
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        degree = torch.LongTensor(degree)
    learning_type = "transductive"
    return adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type

def load_citation(dataset_str="cora", normalization="AugNormAdj", porting_to_torch=True,data_path=datadir, task_type="full"):
    """
    Load Citation Networks Datasets.
    """
    if 'nell' in dataset_str:
        return load_nell("nell.0.001", normalization, porting_to_torch,data_path, task_type)

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str.lower(), names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # degree = np.asarray(G.degree)
    degree = np.sum(adj, axis=1)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if task_type == "full":
        print("Load full supervised task.")
        #supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(ally)- 500)
        idx_val = range(len(ally) - 500, len(ally))
    elif task_type == "semi":
        print("Load semi-supervised task.")
        #semi-supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)
    else:
        raise ValueError("Task type: %s is not supported. Available option: full and semi.")

    adj, features = preprocess_citation(adj, features, normalization)
    features = np.array(features.todense())
    labels = np.argmax(labels, axis=1)
    # porting to pytorch
    if porting_to_torch:
        features = torch.FloatTensor(features).float()
        labels = torch.LongTensor(labels)
        # labels = torch.max(labels, dim=1)[1]
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        degree = torch.LongTensor(degree)
    learning_type = "transductive"
    return adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type

def sgc_precompute(features, adj, degree):
    #t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = 0 #perf_counter()-t
    return features, precompute_time

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)


def loadRedditFromNPZ(dataset_dir=datadir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir +"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def load_reddit_transductive(normalization="AugNormAdj", porting_to_torch=True, data_path=datadir):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ(data_path)
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T + sp.eye(adj.shape[0])
    # train_adj = adj[train_index, :][:, train_index]
    # degree = np.sum(train_adj, axis=1)
    degree = np.sum(adj, axis=1)

    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    # train_features = torch.index_select(features, 0, torch.LongTensor(train_index))
    if not porting_to_torch:
        features = features.numpy()
        # train_features = train_features.numpy()

    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    # train_adj = adj_normalizer(train_adj)

    if porting_to_torch:
        # train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        degree = torch.LongTensor(degree)
        train_index = torch.LongTensor(train_index)
        val_index = torch.LongTensor(val_index)
        test_index = torch.LongTensor(test_index)
    train_adj, train_features = adj, features
    learning_type = "transductive"
    return adj, train_adj, features, train_features, labels, train_index, val_index, test_index, degree, learning_type


def load_reddit_data(normalization="AugNormAdj", porting_to_torch=True, data_path=datadir):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ(data_path)
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T + sp.eye(adj.shape[0])
    train_adj = adj[train_index, :][:, train_index]
    degree = np.sum(train_adj, axis=1)

    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    train_features = torch.index_select(features, 0, torch.LongTensor(train_index))
    if not porting_to_torch:
        features = features.numpy()
        train_features = train_features.numpy()

    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    train_adj = adj_normalizer(train_adj)

    if porting_to_torch:
        train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        degree = torch.LongTensor(degree)
        train_index = torch.LongTensor(train_index)
        val_index = torch.LongTensor(val_index)
        test_index = torch.LongTensor(test_index)
    learning_type = "inductive"
    return adj, train_adj, features, train_features, labels, train_index, val_index, test_index, degree, learning_type




def data_loader(dataset, data_path=datadir, normalization="AugNormAdj", porting_to_torch=True, task_type = "full"):
    if dataset == "reddit":
        # return load_reddit_data(normalization, porting_to_torch, data_path)
        return load_reddit_transductive(normalization, porting_to_torch, data_path)
    else:
        (adj,
         features,
         labels,
         idx_train,
         idx_val,
         idx_test,
         degree,
         learning_type) = load_citation(dataset, normalization, porting_to_torch, data_path, task_type)
        train_adj = adj
        train_features = features
        return adj, train_adj, features, train_features, labels, idx_train, idx_val, idx_test, degree, learning_type



def get_pairwise_sim(x):
    import itertools
    x = x.detach().cpu()
    cos = 0
    n = 0
    indices = [pair for pair in itertools.combinations(range(len(x)), 2)]
    indices = np.array(indices).T
    coser = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim = coser(x[indices[0]], x[indices[1]])
    return torch.abs(cos_sim).mean()


######################################################
def NetMelt(adj, k):
    '''find k edges'''
    u, lambda_1, v = get_dominant_eigen(adj)
    edges = adj.nonzero()
    scores = []
    for n1, n2 in edges:
        score.append([(n1, n2), u[n1]*v[n2]])
    sorted_score = sorted(score, key=lambda x:x[1], reverse=True)
    edges_to_melt = [x[0] for x in sorted_score[:k]]
    return edges_to_melt

def get_dominant_eigen(adj):
    '''calculate dominant eigenvector and eigenvalue'''

    if u.min() < 0:
        u = -u
    if v.min() < 0:
        v = -v
    return u, lambda_1, v

def sample_edges():
    '''sample from unlabelled edges'''
    return sampled_edges, adj

import torch.nn.functional as F
def lploss(output, edges):
    '''link prediction loss'''
    lploss = -torch.sigmoid((output[edges[0]] * output[edges[1]]).mean(1)).sum()
    print(lploss)
    # lploss = 0
    return lploss


import torch
def experiments(embeddings, adj):
    high, mid, low = split_by_degree(adj, thresholds=[2, 15])

    print()
    print('high degrees nodes:', distance(embeddings, high))
    print('mid degree nodes:', distance(embeddings, mid))
    print('low degree nodes:', distance(embeddings, low))
    # distance(embeddings[low], embeddings.mean(0))

    import ipdb
    ipdb.set_trace()

    return

def distance(embeddings, indices):
    return torch.norm(embeddings[indices].mean(0) - embeddings.mean(0)).item()
    # return torch.norm(embeddings - mean)/(embeddings.shape[0] * embeddings.shape[1])

def split_by_degree(adj, thresholds=[1, 10]):
    node_degree = adj.sum(0).A1

    idx = range(len(node_degree))
    degrees = [(i, node_degree[i]) for i in idx]
    degrees = sorted(degrees, key=lambda x:x[1], reverse=True) # descending order by degree
    high = [n for n, d in degrees if d > thresholds[1]]
    mid = [n for n, d in degrees if d > thresholds[0] and d <= thresholds[1]]
    low = [n for n, d in degrees if d <= thresholds[0]]
    k = len(high)
    high_ = np.random.permutation(high)[:k]
    mid_ = np.random.permutation(mid)[:k]
    low_ = np.random.permutation(low)[:k]
    print(f'high degree range from {degrees[0][1]} - {degrees[len(high)-1][1]}, percentage of number of nodes: {len(high_)/len(idx)}')
    print(f'mid degree range from {degrees[len(high)][1]} - {degrees[len(high)+len(mid)-1][1]}, percentage of number of nodes: {len(mid_)/len(idx)}')
    print(f'low degree range from {degrees[len(high)+len(mid)][1]} - {degrees[-1][1]}, percentage of number of nodes: {len(low_)/len(idx)}')

    return high_, mid_, low_


def get_train_val_test_gcn(labels):
    '''
        This setting follows gcn, where we randomly sample 20 instances for each class
        as training data, 500 instances as validation data, 1000 instances as test data.
    '''
    idx = np.arange(len(labels))
    nclass = labels.max() + 1
    idx_train = []
    idx_unlabeled = []
    for i in range(nclass):
        labels_i = idx[labels==i]
        labels_i = np.random.permutation(labels_i)
        idx_train = np.hstack((idx_train, labels_i[: 20])).astype(np.int)
        idx_unlabeled = np.hstack((idx_unlabeled, labels_i[20: ])).astype(np.int)

    idx_unlabeled = np.random.permutation(idx_unlabeled)
    idx_val = idx_unlabeled[: 500]
    idx_test = idx_unlabeled[500: 1500]
    return idx_train, idx_val, idx_test

