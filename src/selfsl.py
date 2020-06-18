import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
import torch
import networkx as nx
from sklearn.cluster import KMeans
from ssl_utils import *
from distance import *
import os


class Base:

    def __init__(self, adj, features, device):
        self.adj = adj
        self.features = features.to(device)
        self.device = device
        self.cached_adj_norm = None

    def get_adj_norm(self):
        if self.cached_adj_norm is None:
            adj_norm = preprocess_adj(self.adj, self.device)
            self.cached_adj_norm= adj_norm
        return self.cached_adj_norm

    def make_loss(self, embeddings):
        return 0

    def transform_data(self):
        return self.get_adj_norm(), self.features

class EdgeMask(Base):

    def __init__(self, adj, features, nhid, device):
        self.adj = adj
        self.masked_edges = None
        self.device = device
        self.features = features.to(device)
        self.cached_adj_norm = None
        self.pseudo_labels = None
        self.linear = nn.Linear(nhid, 2).to(device)

    def transform_data(self, mask_ratio=0.1):
        '''randomly mask edges'''
        # self.cached_adj_norm = None
        if self.cached_adj_norm is None:
            nnz = self.adj.nnz
            perm = np.random.permutation(nnz)
            preserve_nnz = int(nnz*(1 - mask_ratio))
            masked = perm[preserve_nnz: ]
            self.masked_edges = (self.adj.row[masked], self.adj.col[masked])
            perm = perm[:preserve_nnz]
            r_adj = sp.coo_matrix((self.adj.data[perm],
                                   (self.adj.row[perm],
                                    self.adj.col[perm])),
                                  shape=self.adj.shape)

            # renormalize_adj
            r_adj = preprocess_adj(r_adj, self.device)
            self.cached_adj_norm = r_adj
            # features = preprocess_features(self.features, self.device)
        return self.cached_adj_norm, self.features

    def make_loss(self, embeddings):
        '''link prediction loss'''
        edges = self.masked_edges
        # self.neg_edges = self.neg_sample(k=len(edges[0]))
        if self.pseudo_labels is None:
            self.pseudo_labels = np.zeros(2*len(edges[0]))
            self.pseudo_labels[: len(edges[0])] = 1
            self.pseudo_labels = torch.LongTensor(self.pseudo_labels).to(self.device)
            self.neg_edges = self.neg_sample(k=len(edges[0]))

        neg_edges = self.neg_edges
        node_pairs = np.hstack((np.array(edges), np.array(neg_edges).transpose()))
        self.node_pairs = node_pairs
        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]

        embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
        output = F.log_softmax(embeddings, dim=1)

        loss = F.nll_loss(output, self.pseudo_labels)
        # print(loss)
        # from metric import accuracy
        # acc = accuracy(output, self.pseudo_labels)
        # print(acc)
        # # return 0
        return loss

    def neg_sample(self, k):
        nonzero = set(zip(*self.adj.nonzero()))
        edges = self.random_sample_edges(self.adj, k, exclude=nonzero)
        return edges

    def random_sample_edges(self, adj, n, exclude):
        '''
            'exclude' is a set which contains the edges we do not want to sample
             and the edges already sampled
        '''
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]

    def sample_forever(self, adj, exclude):
        '''
            'exclude' is a set which contains the edges we do not want to sample
             and the edges already sampled
        '''
        while True:
            # t = tuple(np.random.randint(0, adj.shape[0], 2))
            t = tuple(random.sample(range(0, adj.shape[0]), 2))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))


class AttributeMask(Base):

    def __init__(self, adj, features, idx_train, nhid, device):
        self.adj = adj
        self.cached_adj_norm = None
        self.features = features.to(device)
        self.nfeat = features.shape[1]
        self.device = device
        self.masked_indicator = torch.zeros(self.nfeat).to(device)
        self.masked_nodes = None
        self.linear = nn.Linear(nhid, self.nfeat).to(device)
        self.cached_features = None
        self.labeled = idx_train.cpu().numpy()
        self.all = np.arange(adj.shape[0])
        self.unlabeled = np.array([n for n in self.all if n not in idx_train])

    def reset(self):
        self.cached_features = None

    def transform_data(self, mask_ratio=0.1):
        # self.cached_features = None
        if self.cached_features is None:
            # nnodes = self.adj.shape[0]
            # perm = np.random.permutation(nnodes)
            perm = np.random.permutation(self.unlabeled)
            self.masked_nodes = perm[: int(len(perm)*(mask_ratio))]
            self.cached_features = self.features.clone()
            self.cached_features[self.masked_nodes] = self.masked_indicator
            self.pseudo_labels = self.features[self.masked_nodes]
            # self.pseudo_labels[self.pseudo_labels > 0] = 1
        return self.get_adj_norm(), self.cached_features

    def make_loss(self, embeddings):
        masked_embeddings = (self.linear(embeddings[self.masked_nodes]))
        loss = F.mse_loss(masked_embeddings, self.pseudo_labels, reduction='mean')
        # print(loss)
        return loss


class NodeProperty(Base):

    def __init__(self, adj, features, nhid, device, regression=False):
        self.adj = adj
        self.features = features.to(device)
        self.nfeat = features.shape[1]
        self.cached_adj_norm = None
        self.device = device
        self.adj_nx = nx.from_scipy_sparse_matrix(self.adj)
        self.regression = regression
        if regression:
            self.linear = nn.Linear(nhid, 1).to(device)
        else:
            self.linear = nn.Linear(nhid, 3).to(device)

        self.metric_to_func = {'pagerank': calc_pagerank,
                               'degree': calc_degree,
                               'centrality': calc_centrality,
                               'node_importance': calc_node_importance,
                               'clustering_coeff': calc_clustering_coeff}
        self.pseudo_labels = None

    def transform_data(self):
        return self.get_adj_norm(), self.features

    def make_loss(self, embeddings, metric='degree'):
        if self.regression:
            return self.regression_loss(embeddings, metric)
        else:
            return self.classification_loss(embeddings, metric)

    def regression_loss(self, embeddings, metric):
        if self.pseudo_labels is None:
            self.pseudo_labels = self.metric_to_func[metric](self.adj_nx, self.device, multi_class=False).view(-1,1)

        embeddings = (self.linear(embeddings))
        loss = F.mse_loss(embeddings, self.pseudo_labels, reduction='mean')
        # print(loss)
        return loss

    def classification_loss(self, embeddings, metric):
        if self.pseudo_labels is None:
            self.pseudo_labels = self.metric_to_func[metric](self.adj_nx, self.device, multi_class=True)
            self.sampled_indices = (self.pseudo_labels >= 0)
        embeddings = self.linear(embeddings)

        output = F.log_softmax(embeddings, dim=1)
        loss = F.nll_loss(output[self.sampled_indices], self.pseudo_labels[self.sampled_indices])
        # from metric import accuracy
        # acc = accuracy(output[self.sampled_indices], self.pseudo_labels[self.sampled_indices])
        # print(acc)
        return loss

class PairwiseDistance(Base):

    def __init__(self, adj, features, nhid, device, idx_train, regression=False):
        self.adj = adj
        self.features = features.to(device)
        self.nfeat = features.shape[1]
        self.cached_adj_norm = None
        self.device = device

        self.labeled = idx_train.cpu().numpy()
        self.all = np.arange(adj.shape[0])
        self.unlabeled = np.array([n for n in self.all if n not in idx_train])

        self.regression = regression
        self.nclass = 4
        if regression:
            self.linear = nn.Linear(nhid, self.nclass).to(device)
        else:
            self.linear = nn.Linear(nhid, self.nclass).to(device)

        self.pseudo_labels = None

    def transform_data(self):
        return self.get_adj_norm(), self.features

    def make_loss(self, embeddings):
        if self.regression:
            return self.regression_loss(embeddings)
        else:
            return self.classification_loss(embeddings)

    def classification_loss(self, embeddings):
        if self.pseudo_labels is None:
            self.agent = NodeDistance(self.adj, nclass=self.nclass)
            self.pseudo_labels = self.agent.get_label().to(self.device)

        # embeddings = F.dropout(embeddings, 0, training=True)
        self.node_pairs = self.sample(self.agent.distance)
        node_pairs = self.node_pairs
        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]

        embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
        output = F.log_softmax(embeddings, dim=1)
        loss = F.nll_loss(output, self.pseudo_labels[node_pairs])
        print(loss)
        # from metric import accuracy
        # acc = accuracy(output, self.pseudo_labels[node_pairs])
        # print(acc)
        return loss

    def sample(self, labels, ratio=0.1, k=4000):
        node_pairs = []
        for i in range(1, labels.max()+1):
            tmp = np.array(np.where(labels==i)).transpose()
            indices = np.random.choice(np.arange(len(tmp)), k, replace=False)
            node_pairs.append(tmp[indices])
        node_pairs = np.array(node_pairs).reshape(-1, 2).transpose()
        return node_pairs[0], node_pairs[1]

    def _sample(self, labels, ratio=0.1, k=400):
        # first sample k nodes
        candidates = self.all
        # perm = np.random.choice(candidates, int(ratio*len(candidates)), replace=False)
        perm = np.random.choice(candidates, 300, replace=False)

        node_pairs = []
        # then sample k other nodes to make sure class balance
        for i in range(1, labels.max()+1):
            tmp = np.array(np.where(labels==i)).transpose()
            tmp_0 = tmp[:, 0]
            targets = np.where(tmp_0.reshape(tmp_0.size, 1) == perm)[0]
            # targets = np.array([True if x in perm else False for x in tmp[:, 0]])
            # indices = np.random.choice(np.arange(len(tmp))[targets], k, replace=False)
            indices = np.random.choice(targets, k, replace=False)
            node_pairs.append(tmp[indices])
        node_pairs = np.array(node_pairs).reshape(-1, 2).transpose()
        return node_pairs[0], node_pairs[1]


class DistanceCluster(Base):

    def __init__(self, adj, features, nhid, device, idx_train, args, regression=True):
        self.adj = adj
        self.features = features.to(device)
        self.nfeat = features.shape[1]
        self.cached_adj_norm = None
        self.device = device

        self.labeled = idx_train.cpu().numpy()
        self.all = np.arange(adj.shape[0])
        self.unlabeled = np.array([n for n in self.all if n not in idx_train])

        self.regression = regression
        d = {'pubmed': 30, 'cora':10, 'citeseer': 10}
        self.cluster_number = d[args.dataset]
        if regression:
            self.linear = nn.Linear(nhid, self.cluster_number).to(device)
        else:
            self.linear = nn.Linear(nhid, self.cluster_number).to(device)
        self.pseudo_labels = None

    def transform_data(self):
        return self.get_adj_norm(), self.features

    def make_loss(self, embeddings):
        if self.regression:
            return self.regression_loss(embeddings)
        else:
            return self.classification_loss(embeddings)

    def regression_loss(self, embeddings):
        if self.pseudo_labels is None:
            cluster_agent = ClusteringMachine(self.adj, self.features, self.cluster_number)
            cluster_agent.decompose()
            self.pseudo_labels = cluster_agent.dis_matrix.to(self.device)

        embeddings = (self.linear(embeddings))
        loss = F.mse_loss(embeddings, self.pseudo_labels, reduction='mean')

        # loss = torch.norm(embeddings - self.pseudo_labels)
        # print(loss)
        # loss = 0
        return loss

    def classification_loss(self, embeddings):
        """
        Predict the closest cluster
        """
        if self.pseudo_labels is None:
            cluster_agent = ClusteringMachine(self.adj, self.cluster_number)
            cluster_agent.decompose()
            # self.pseudo_labels = cluster_agent.dis_matrix.to(self.device).argmin(1)
            self.pseudo_labels = cluster_agent.dis_matrix.to(self.device)
            self.pseudo_labels[self.pseudo_labels < 8] = 0
            self.pseudo_labels[self.pseudo_labels >= 8] = 1


        embeddings = (self.linear(embeddings))
        # output = F.log_softmax(embeddings, dim=1)
        # loss = F.nll_loss(output, self.pseudo_labels)
        criterion = nn.MultiLabelSoftMarginLoss()
        loss = criterion(embeddings[self.unlabeled], self.pseudo_labels[self.unlabeled])
        print(loss)
        return loss


class ICAContextLabel(Base):

    def __init__(self, adj, features, labels, nhid, nclass, device, idx_train, args, regression=False):
        self.adj = adj
        try:
            self.sp_features = sp.csr_matrix(features.numpy())
        except:
            self.sp_features = to_scipy(features)
        self.features = features.to(device)
        self.labels = encode_onehot(labels)
        self.args = args

        self.nfeat = features.shape[1]
        self.cached_adj_norm = None
        self.device = device

        self.idx_train = idx_train
        self.labeled = idx_train.cpu().numpy()
        self.all = np.arange(adj.shape[0])
        self.unlabeled = np.array([n for n in self.all if n not in idx_train])

        self.regression = regression
        if regression:
            self.linear = nn.Linear(nhid, 1).to(device)
        else:
            self.linear = nn.Linear(nhid, nclass).to(device)
        self.pseudo_labels = None
        self.label_correction = False
        self.labels_corrected = None
        self.verbose = True

    def transform_data(self):
        if self.pseudo_labels is None:
            self.agent = ICAAgent(self.adj, self.sp_features, self.labels, self.labeled, self.unlabeled, args=self.args)
            self.pseudo_labels = self.agent.get_label().to(self.device)
            # self.pseudo_labels = self.agent.get_label(use_probs=True).to(self.device)
        return self.get_adj_norm(), self.features

    def make_loss(self, embeddings):
        if self.regression:
            return self.regression_loss(embeddings)
        else:
            return self.classification_loss(embeddings)

    def classification_loss(self, embeddings):
        if self.label_correction:
            if self.args.dataset == 'cora':
                n_samples = 1000 # 2000
            if self.args.dataset == 'citeseer':
                n_samples = 3000 # 1000
            if self.args.dataset == 'pubmed':
                n_samples = 10000
            idx_sampled = np.random.choice(self.unlabeled,
                                n_samples, replace=False)
            corrected = self.dist_corrected(embeddings, idx_sampled)

        embeddings = F.dropout(embeddings, 0.5, training=True)
        embeddings = self.linear(embeddings)
        output = torch.softmax(embeddings, dim=1)
        # output = F.log_softmax(embeddings, dim=1)
        # loss = F.nll_loss(output[self.unlabeled], self.pseudo_labels[self.unlabeled])
        loss = F.mse_loss(output[self.unlabeled], self.pseudo_labels[self.unlabeled], reduction='mean')
        if self.label_correction:
            # loss_corrected = F.mse_loss(output[self.unlabeled], corrected[self.unlabeled])
            loss_corrected = F.mse_loss(output[idx_sampled], corrected[idx_sampled])
            alpha = self.args.alpha
            # if self.args.dataset == 'cora':
            #     alpha = 1.5
            # if self.args.dataset == 'citeseer':
            #     alpha = 0.8
            # if self.args.dataset == 'pubmed':
            #     alpha = 1
            # if self.args.dataset == 'reddit':
            #     alpha = 1
            loss = loss + alpha * loss_corrected
        if self.verbose:
            print(loss)
            from metric import accuracy
            acc = accuracy(output[self.unlabeled], self.pseudo_labels.argmax(1)[self.unlabeled])
            # acc = accuracy(output[self.unlabeled], self.pseudo_labels[self.unlabeled])
            print(acc)
        return loss

    def dist_corrected(self, embeddings, idx_sampled):
        from ssl_utils import label_correction
        self.labels_corrected = None
        if self.labels_corrected is None:
            self.labels_corrected = self.agent.concated
            acc = (self.labels.argmax(1) == self.agent.concated)[idx_sampled].sum()/ len(self.agent.concated[idx_sampled])
            if self.verbose:
                print('corrected acc: %s' % acc)
        corrected = label_correction(embeddings.detach(), self.labels_corrected, idx_sampled, idx_train=None, is_numpy=True)
        labels_corrected = np.copy(self.agent.concated)
        labels_corrected[idx_sampled] = corrected
        # acc = (self.labels.argmax(1) == labels_corrected)[self.unlabeled].sum()/ len(labels_corrected)
        if self.verbose:
            acc = (self.labels.argmax(1) == labels_corrected)[idx_sampled].sum()/ len(labels_corrected[idx_sampled])
            print('corrected acc: %s' % acc)
        self.labels_corrected = labels_corrected
        return self.agent.get_label(labels_corrected).to(self.device)


class LPContextLabel(ICAContextLabel):

    def __init__(self, adj, features, labels, nhid, nclass, device, idx_train, args, regression=False):
        super().__init__(adj, features, labels, nhid, nclass, device, idx_train, args, regression)

    def transform_data(self):
        if self.pseudo_labels is None:
            self.agent = LPAgent(self.adj, self.sp_features, self.labels, self.labeled, self.unlabeled, self.args)
            # self.agent = FeatLPAgent(self.adj, self.sp_features, self.labels, self.labeled, self.unlabeled, self.args)
            self.pseudo_labels = self.agent.get_label().to(self.device)
        return self.get_adj_norm(), self.features

class CombinedContextLabel(ICAContextLabel):

    def __init__(self, adj, features, labels, nhid, nclass, device, idx_train, args, regression=False):
        super().__init__(adj, features, labels, nhid, nclass, device, idx_train, args, regression)

    def transform_data(self):
        if self.pseudo_labels is None:
            self.agent = CombinedAgent(self.adj, self.sp_features, self.labels, self.labeled, self.unlabeled, args=self.args)
            self.pseudo_labels = self.agent.get_label().to(self.device)
        return self.get_adj_norm(), self.features


class Distance2Labeled(Base):

    def __init__(self, adj, features, labels, nhid, nclass, device, idx_train, regression=True):
        self.adj = adj
        self.features = features.to(device)
        self.nfeat = features.shape[1]
        self.cached_adj_norm = None
        self.device = device

        self.labeled = idx_train.cpu().numpy()
        self.all = np.arange(adj.shape[0])
        self.unlabeled = np.array([n for n in self.all if n not in idx_train])

        self.node2label = {}
        for x in self.labeled:
            if labels[x] not in self.node2label:
                self.node2label[labels[x]] = []
            self.node2label[labels[x]].append(x)

        self.regression = regression
        self.nclass = nclass
        # self.nclass = 3 * nclass
        if regression:
            self.linear = nn.Linear(nhid, self.nclass).to(device)
        else:
            self.linear = nn.Linear(nhid, self.nclass).to(device)
        self.pseudo_labels = None

    def transform_data(self):
        return self.get_adj_norm(), self.features

    def make_loss(self, embeddings):
        if self.regression:
            return self.regression_loss(embeddings)
        else:
            return self.classification_loss(embeddings)

    def regression_loss(self, embeddings):
        if self.pseudo_labels is None:
            agent = LabeledNodeDistance(self.adj, self.node2label)
            self.pseudo_labels = agent.get_label().to(self.device)

        embeddings = (self.linear(embeddings))

        # loss = F.mse_loss(embeddings, self.pseudo_labels, reduction='mean')
        loss = F.mse_loss(embeddings[self.unlabeled], self.pseudo_labels[self.unlabeled], reduction='mean')

        # print(loss)
        # loss = 0
        return loss

    def classification_loss(self, embeddings):
        """
        Predict the closest cluster
        """
        if self.pseudo_labels is None:
            cluster_agent = ClusteringMachine(self.adj, self.cluster_number)
            cluster_agent.decompose()
            # self.pseudo_labels = cluster_agent.dis_matrix.to(self.device).argmin(1)
            self.pseudo_labels = cluster_agent.dis_matrix.to(self.device)
            self.pseudo_labels[self.pseudo_labels < 8] = 0
            self.pseudo_labels[self.pseudo_labels >= 8] = 1

        embeddings = (self.linear(embeddings))
        # output = F.log_softmax(embeddings, dim=1)
        # loss = F.nll_loss(output, self.pseudo_labels)
        criterion = nn.MultiLabelSoftMarginLoss()
        loss = criterion(embeddings[self.unlabeled], self.pseudo_labels[self.unlabeled])
        # print(loss)
        return loss


def preprocess_features(features, device):
    return features.to(device)

def preprocess_adj(adj, device):
    # adj_normalizer = fetch_normalization(normalization)
    adj_normalizer = aug_normalized_adjacency
    r_adj = adj_normalizer(adj)
    r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
    r_adj = r_adj.to(device)
    return r_adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

class PairwiseAttrSim(Base):

    def __init__(self, adj, features, nhid, device, idx_train, args, regression=True):
        self.adj = adj
        self.args = args
        self.features = features.to(device)
        self.nfeat = features.shape[1]
        self.cached_adj_norm = None
        self.device = device

        self.labeled = idx_train.cpu().numpy()
        self.all = np.arange(adj.shape[0])
        self.unlabeled = np.array([n for n in self.all if n not in idx_train])

        self.regression = regression
        self.nclass = 1
        if regression:
            self.linear = nn.Linear(nhid, self.nclass).to(device)
        else:
            self.linear = nn.Linear(nhid, self.nclass).to(device)

        self.pseudo_labels = None

    def transform_data(self):
        return self.get_adj_norm(), self.features

    def make_loss(self, embeddings):
        if self.regression:
            return self.regression_loss(embeddings)
        else:
            return self.classification_loss(embeddings)

    def regression_loss(self, embeddings):
        if self.pseudo_labels is None:
            agent = AttrSim(self.features, self.args)
            self.pseudo_labels = agent.get_label().to(self.device)
            # self.pseudo_labels = agent.get_label()
            self.node_pairs = agent.node_pairs
            # self.idx = np.arange(len(self.node_pairs[0]))

        k = 5000
        sampled = np.random.choice(len(self.node_pairs[0]), k, replace=False)

        node_pairs = self.node_pairs
        embeddings0 = embeddings[node_pairs[0][sampled]]
        embeddings1 = embeddings[node_pairs[1][sampled]]

        embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
        loss = F.mse_loss(embeddings, self.pseudo_labels[sampled], reduction='mean')
        # print(loss)
        return loss

    def classification_loss(self, embeddings):
        if self.pseudo_labels is None:
            self.agent = NodeDistance(self.adj, nclass=self.nclass)
            self.pseudo_labels = self.agent.get_label().to(self.device)

        # embeddings = F.dropout(embeddings, 0, training=True)
        self.node_pairs = self.sample(self.agent.distance)
        node_pairs = self.node_pairs
        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]

        embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
        output = F.log_softmax(embeddings, dim=1)
        loss = F.nll_loss(output, self.pseudo_labels[node_pairs])
        # print(loss)
        # from metric import accuracy
        # acc = accuracy(output, self.pseudo_labels[node_pairs])
        # print(acc)
        return loss

    def sample(self, labels, ratio=0.1, k=4000):
        node_pairs = []
        for i in range(1, labels.max()+1):
            tmp = np.array(np.where(labels==i)).transpose()
            indices = np.random.choice(np.arange(len(tmp)), k, replace=False)
            node_pairs.append(tmp[indices])
        node_pairs = np.array(node_pairs).reshape(-1, 2).transpose()
        return node_pairs[0], node_pairs[1]


