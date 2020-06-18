# coding=utf-8
import numpy as np
import torch
import scipy.sparse as sp
from utils import data_loader, sparse_mx_to_torch_sparse_tensor
from normalization import fetch_normalization
from sklearn.decomposition import PCA, TruncatedSVD

class Sampler:
    """Sampling the input graph data."""
    def __init__(self, dataset, args, data_path="data", task_type="full"):
        self.dataset = dataset
        self.data_path = data_path
        (self.adj,
         self.train_adj,
         self.features,
         self.train_features,
         self.labels,
         self.idx_train,
         self.idx_val,
         self.idx_test,
         self.degree,
         self.learning_type) = data_loader(dataset, data_path, "NoNorm", False, task_type)

        #convert some data to torch tensor ---- may be not the best practice here.

        self.train_adj = self.adj
        self.train_features = self.features
        self.learning_type = 'transductive'

        self.labels = self.labels.astype(np.int)
        if args.pca:
            # pca = TruncatedSVD(n_components=600, n_iter=5000, algorithm='arpack')
            pca = PCA(n_components=256)
            self.features = pca.fit_transform(self.features)
            self.train_features = pca.fit_transform(self.train_features)

        self.features = torch.FloatTensor(self.features).float()
        self.train_features = torch.FloatTensor(self.train_features).float()
        # self.train_adj = self.train_adj.tocsr()


        if args.train_size and not args.fastmode:
            from ssl_utils import get_splits_each_class
            self.idx_train, self.idx_val, self.idx_test = get_splits_each_class(
                    labels=self.labels, train_size=args.train_size)
            # print(self.idx_train[:10])
            # from ssl_utils import get_few_labeled_splits
            # self.idx_train, self.idx_val, self.idx_test = get_few_labeled_splits(
            #         labels=self.labels, train_size=args.train_size)

        if args.fastmode:
            from deeprobust.graph.utils import get_train_test
            self.idx_train, self.idx_test = get_train_test(
                    nnodes=self.adj.shape[0], test_size=1-args.label_rate,
                    stratify=self.labels)
            self.idx_test = self.idx_test[:1000]

        self.labels_torch = torch.LongTensor(self.labels)
        self.idx_train_torch = torch.LongTensor(self.idx_train)
        self.idx_val_torch = torch.LongTensor(self.idx_val)
        self.idx_test_torch = torch.LongTensor(self.idx_test)
        # vertex_sampler cache
        # where return a tuple
        self.pos_train_idx = np.where(self.labels[self.idx_train] == 1)[0]
        self.neg_train_idx = np.where(self.labels[self.idx_train] == 0)[0]
        # self.pos_train_neighbor_idx = np.where

        self.nfeat = self.features.shape[1]
        self.nclass = int(self.labels.max().item() + 1)
        self.trainadj_cache = {}
        self.adj_cache = {}
        #print(type(self.train_adj))
        self.degree_p = None

    def _preprocess_adj(self, normalization, adj, cuda):
        adj_normalizer = fetch_normalization(normalization)
        r_adj = adj_normalizer(adj)
        r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
        if cuda:
            r_adj = r_adj.cuda()
        return r_adj

    def _preprocess_fea(self, fea, cuda):
        if cuda:
            return fea.cuda()
        else:
            return fea

    def stub_sampler(self, normalization, cuda):
        """
        The stub sampler. Return the original data.
        """
        if normalization in self.trainadj_cache:
            r_adj = self.trainadj_cache[normalization]
        else:
            r_adj = self._preprocess_adj(normalization, self.train_adj, cuda)
            self.trainadj_cache[normalization] = r_adj
        fea = self._preprocess_fea(self.train_features, cuda)

        # r_adj = torch.eye(r_adj.shape[0]).cuda()
        return r_adj, fea

    def randomedge_sampler(self, percent, normalization, cuda):
        """
        Randomly drop edge and preserve percent% edges.
        """
        "Opt here"
        if percent >= 1.0:
            return self.stub_sampler(normalization, cuda)

        nnz = self.train_adj.nnz
        perm = np.random.permutation(nnz)
        preserve_nnz = int(nnz*percent)
        perm = perm[:preserve_nnz]
        r_adj = sp.coo_matrix((self.train_adj.data[perm],
                               (self.train_adj.row[perm],
                                self.train_adj.col[perm])),
                              shape=self.train_adj.shape)
        r_adj = self._preprocess_adj(normalization, r_adj, cuda)
        fea = self._preprocess_fea(self.train_features, cuda)
        return r_adj, fea

    def vertex_sampler(self, percent, normalization, cuda):
        """
        Randomly drop vertexes.
        """
        if percent >= 1.0:
            return self.stub_sampler(normalization, cuda)
        self.learning_type = "inductive"
        pos_nnz = len(self.pos_train_idx)
        # neg_neighbor_nnz = 0.4 * percent
        neg_no_neighbor_nnz = len(self.neg_train_idx)
        pos_perm = np.random.permutation(pos_nnz)
        neg_perm = np.random.permutation(neg_no_neighbor_nnz)
        pos_perseve_nnz = int(0.9 * percent * pos_nnz)
        neg_perseve_nnz = int(0.1 * percent * neg_no_neighbor_nnz)
        # print(pos_perseve_nnz)
        # print(neg_perseve_nnz)
        pos_samples = self.pos_train_idx[pos_perm[:pos_perseve_nnz]]
        neg_samples = self.neg_train_idx[neg_perm[:neg_perseve_nnz]]
        all_samples = np.concatenate((pos_samples, neg_samples))
        r_adj = self.train_adj
        r_adj = r_adj[all_samples, :]
        r_adj = r_adj[:, all_samples]
        r_fea = self.train_features[all_samples, :]
        # print(r_fea.shape)
        # print(r_adj.shape)
        # print(len(all_samples))
        r_adj = self._preprocess_adj(normalization, r_adj, cuda)
        r_fea = self._preprocess_fea(r_fea, cuda)
        return r_adj, r_fea, all_samples

    def degree_sampler(self, percent, normalization, cuda):
        """
        Randomly drop edge wrt degree (high degree, low probility).
        """
        if percent >= 0:
            return self.stub_sampler(normalization, cuda)
        if self.degree_p is None:
            degree_adj = self.train_adj.multiply(self.degree)
            self.degree_p = degree_adj.data / (1.0 * np.sum(degree_adj.data))
        # degree_adj = degree_adj.multi degree_adj.sum()
        nnz = self.train_adj.nnz
        preserve_nnz = int(nnz * percent)
        perm = np.random.choice(nnz, preserve_nnz, replace=False, p=self.degree_p)
        r_adj = sp.coo_matrix((self.train_adj.data[perm],
                               (self.train_adj.row[perm],
                                self.train_adj.col[perm])),
                              shape=self.train_adj.shape)
        r_adj = self._preprocess_adj(normalization, r_adj, cuda)
        fea = self._preprocess_fea(self.train_features, cuda)
        return r_adj, fea


    def get_test_set(self, normalization, cuda):
        """
        Return the test set.
        """
        if self.learning_type == "transductive":
            return self.stub_sampler(normalization, cuda)
        else:
            if normalization in self.adj_cache:
                r_adj = self.adj_cache[normalization]
            else:
                r_adj = self._preprocess_adj(normalization, self.adj, cuda)
                self.adj_cache[normalization] = r_adj
            fea = self._preprocess_fea(self.features, cuda)
            return r_adj, fea

    def get_val_set(self, normalization, cuda):
        """
        Return the validataion set. Only for the inductive task.
        Currently behave the same with get_test_set
        """
        return self.get_test_set(normalization, cuda)

    def get_label_and_idxes(self, cuda):
        """
        Return all labels and indexes.
        """
        if cuda:
            return self.labels_torch.cuda(), self.idx_train_torch.cuda(), self.idx_val_torch.cuda(), self.idx_test_torch.cuda()
        return self.labels_torch, self.idx_train_torch, self.idx_val_torch, self.idx_test_torch



###############

    def independent_labeled(self, percent, normalization, cuda):
        """
        Randomly drop edge and preserve percent% edges.
        """
        "Opt here"
        labeled = self.idx_train
        new_adj = self.train_adj.tolil()

        new_adj[np.ix_(labeled, labeled)] = 0
        new_adj[:, labeled] = 0

        print(new_adj.shape)
        r_adj = new_adj.tocoo()
        # r_adj = sp.coo_matrix((self.train_adj.data[perm],
        #                        (self.train_adj.row[perm],
        #                         self.train_adj.col[perm])),
        #                       shape=self.train_adj.shape)
        r_adj = self._preprocess_adj(normalization, r_adj, cuda)
        fea = self._preprocess_fea(self.train_features, cuda)
        return r_adj, fea




