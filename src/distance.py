import metis
import torch
import random
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import collections
from ica.utils import load_data, pick_aggregator, create_map, build_graph
from ica.classifiers import LocalClassifier, RelationalClassifier, ICA
from scipy.stats import sem
from sklearn.metrics import accuracy_score
import sklearn
from ssl_utils import encode_onehot
from sklearn_extra.cluster import KMedoids
from utils import row_normalize
import numpy as np
import numba
from numba import njit
import os
import time


class ClusteringMachine(object):
    """
    Clustering the graph, feature set and target.
    """
    def __init__(self, adj, features, cluster_number=20, clustering_method='metis'):
        """
        :param graph: Networkx Graph.
        """
        self.adj = adj
        self.features = features.cpu().numpy()
        self.graph = nx.from_scipy_sparse_matrix(adj)
        self.clustering_method = clustering_method
        self.cluster_number = cluster_number

    def decompose(self):
        """
        Decomposing the graph, partitioning, creating Torch arrays.
        """
        if self.clustering_method == "metis":
            print("\nMetis graph clustering started.\n")
            self.metis_clustering()
            central_nodes = self.get_central_nodes()
            self.shortest_path_to_clusters(central_nodes)
        elif self.clustering_method == "kmedoids":
            print("\nKMedoids node clustering started.\n")
            central_nodes = self.kmedoids_clustering()
            self.shortest_path_to_clusters(central_nodes)
        elif self.clustering == "random":
            print("\nRandom graph clustering started.\n")
            self.random_clustering()
            central_nodes = self.get_central_nodes()
            self.shortest_path_to_clusters(central_nodes)

        self.dis_matrix = torch.FloatTensor(self.dis_matrix)
        # self.transfer_edges_and_nodes()

    def random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = [cluster for cluster in range(self.cluster_number)]
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.graph.nodes()}

    def metis_clustering(self):
        """
        Clustering the graph with Metis. For details see:
        """
        (st, parts) = metis.part_graph(self.graph, self.cluster_number)
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def kmedoids_clustering(self):
        from sklearn.decomposition import PCA, TruncatedSVD
        # pca = TruncatedSVD(n_components=600, n_iter=5000, algorithm='arpack')
        # pca = PCA(n_components=256)
        # self.features_pca = pca.fit_transform(self.features)
        # from sklearn import preprocessing
        # scaler = preprocessing.StandardScaler()
        # self.features_pca = scaler.fit_transform(self.features_pca)
        self.features_pca = self.features
        kmedoids = KMedoids(n_clusters=self.cluster_number,
                random_state=0).fit(self.features_pca)
        self.clusters = list(set(kmedoids.labels_))
        self.cluster_membership = kmedoids.labels_

        return self.find_kmedoids_center(kmedoids.cluster_centers_)

    def find_kmedoids_center(self, centers_ori):
        centers = []
        for ii, x in enumerate(self.features_pca):
            # if self.features.shape[1] in (centers_ori == x).sum(1):
            idx = np.where((centers_ori == x).sum(1) == self.features_pca.shape[1])[0]
            if len(idx):
                centers_ori[idx, :] = -1
                centers.append(ii)

        assert len(centers) == self.cluster_number, "duplicate features"
        return centers

    def general_data_partitioning(self):
        """
        Creating data partitions and train-test splits.
        """
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        for cluster in self.clusters:
            subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster])
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] +  [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
            self.sg_train_nodes[cluster], self.sg_test_nodes[cluster] = train_test_split(list(mapper.values()), test_size = 0.8)
            self.sg_test_nodes[cluster] = sorted(self.sg_test_nodes[cluster])
            self.sg_train_nodes[cluster] = sorted(self.sg_train_nodes[cluster])
        print('Number of nodes in clusters:', {x: len(y) for x,y in self.sg_nodes.items()})

    def get_central_nodes(self):
        """
        set the central node as the node with highest degree in the cluster
        """
        self.general_data_partitioning()
        central_nodes = {}
        for cluster in self.clusters:
            counter = {}
            for node, _ in self.sg_edges[cluster]:
                counter[node] = counter.get(node, 0) + 1
            sorted_counter = sorted(counter.items(), key=lambda x:x[1])
            central_nodes[cluster] = sorted_counter[-1][0]
        return central_nodes

    def transfer_edges_and_nodes(self):
        """
        Transfering the data to PyTorch format.
        """
        for cluster in self.clusters:
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster]).t()
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster])
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster])

    def transform_depth(self, depth):
        return 1 / depth

    def shortest_path_to_clusters(self, central_nodes, transform=True):
        """
        Do BFS on each central node, then we can get a node set for each cluster
        which is within k-hop neighborhood of the cluster.
        """
        # self.distance = {c:{} for c in self.clusters}
        self.dis_matrix = -np.ones((self.adj.shape[0], self.cluster_number))
        for cluster in self.clusters:
            node_cur = central_nodes[cluster]
            visited = set([node_cur])
            q = collections.deque([(x, 1) for x in self.graph.neighbors(node_cur)])
            while q:
                node_cur, depth = q.popleft()
                if node_cur in visited:
                    continue
                visited.add(node_cur)
                # if depth not in self.distance[cluster]:
                #     self.distance[cluster][depth] = []
                # self.distance[cluster][depth].append(node_cur)
                if transform:
                    self.dis_matrix[node_cur][cluster] = self.transform_depth(depth)
                else:
                    self.dis_matrix[node_cur][cluster] = depth
                for node_next in self.graph.neighbors(node_cur):
                    q.append((node_next, depth+1))

        if transform:
            self.dis_matrix[self.dis_matrix==-1] = 0
        else:
            self.dis_matrix[self.dis_matrix==-1] = self.dis_matrix.max() + 2
        return self.dis_matrix

    def dfs(self, node_cur, visited, path, cluster_id):
        for node_next in self.graph.neighbors(node_cur):
            if node_next in visited:
                continue
            visited.add(node_next)
            if path not in self.distance[cluster_id]:
                self.distance[cluster_id][path] = []
            self.distance[cluster_id][path].append(node_next)
            self.bfs(node_next, visited, path+1, cluster_id)


class NodeDistance:

    def __init__(self, adj, nclass=4):
        """
        :param graph: Networkx Graph.
        """
        self.adj = adj
        self.graph = nx.from_scipy_sparse_matrix(adj)
        self.nclass = nclass

    def get_label(self):
        path_length = dict(nx.all_pairs_shortest_path_length(self.graph, cutoff=self.nclass-1))
        distance = - np.ones((len(self.graph), len(self.graph))).astype(int)

        for u, p in path_length.items():
            for v, d in p.items():
                distance[u][v] = d

        distance[distance==-1] = distance.max() + 1
        distance = np.triu(distance)
        self.distance = distance
        return torch.LongTensor(distance) - 1

    def _get_label(self):
        '''
        group 1,2 into the same category, 3, 4, 5 separately
        designed for 2-layer GCN
        '''
        path_length = dict(nx.all_pairs_shortest_path_length(self.graph, cutoff=self.nclass))
        distance = - np.ones((len(self.graph), len(self.graph))).astype(int)

        for u, p in path_length.items():
            for v, d in p.items():
                distance[u][v] = d

        distance[distance==-1] = distance.max() + 1

        # group 1, 2 in to one category
        distance = np.triu(distance)
        distance[distance==1] = 2
        self.distance = distance - 1
        return torch.LongTensor(distance) - 2

    def sample(self, labels, ratio=0.1):
        # first sample k nodes
        # candidates = self.all
        candidates = np.arange(len(self.graph))
        perm = np.random.choice(candidates, int(ratio*len(candidates)), replace=False)
        # then sample k other nodes to make sure class balance
        node_pairs = []
        for i in range(1, labels.max()+1):
            tmp = np.array(np.where(labels==i)).transpose()
            indices = np.random.choice(np.arange(len(tmp)), 10, replace=False)
            node_pairs.append(tmp[indices])
        node_pairs = np.array(node_pairs).reshape(-1, 2).transpose()
        return node_pairs[0], node_pairs[1]


class LabeledNodeDistance:

    def __init__(self, adj, node2label):
        """
        Average Distance to labeled node in each class
        """
        self.adj = adj
        self.graph = nx.from_scipy_sparse_matrix(adj)
        self.nclass = max(node2label.keys()) + 1
        self.node2label = node2label
        self.shortest_path_to_nodes()

    def get_label(self):
        return torch.FloatTensor(self.dis_matrix)

    def transform_depth(self, depth):
        return 1 / depth
        # level = [1, 2, 3, 4]
        # for i in range(0, len(level)-1):
        #     cond = ((depth >= level[i]) and (depth < level[i+1]))
        #     if cond = True:
        #         return
        #     break

    def _shortest_path_to_nodes(self):
        """
        Do BFS from each labeled node, then we can get the shortest path length
        from other nodes to each labeled node
        """
        self.dis_matrix = -np.ones((self.adj.shape[0], self.nclass * 3))
        for c, nodes in self.node2label.items():
            dis_matrix_ind = -np.ones((self.adj.shape[0], len(nodes)))
            for ii, node_st in enumerate(nodes):
                # do bfs
                visited = set([node_st])
                q = collections.deque([(x, 1) for x in self.graph.neighbors(node_st)])
                while q:
                    node_cur, depth = q.popleft()
                    if node_cur in visited:
                        continue
                    visited.add(node_cur)
                    dis_matrix_ind[node_cur][ii] = self.transform_depth(depth)
                    for node_next in self.graph.neighbors(node_cur):
                        q.append((node_next, depth+1))
            # dis_matrix_ind[dis_matrix_ind==-1] = dis_matrix_ind.max() + 2
            dis_matrix_ind[dis_matrix_ind==-1] = 0
            self.dis_matrix[:, c*3] = dis_matrix_ind.mean(1)
            self.dis_matrix[:, c*3+1] = dis_matrix_ind.max(1)
            self.dis_matrix[:, c*3+2] = dis_matrix_ind.min(1)

    def shortest_path_to_nodes(self):
        """
        Do BFS from each labeled node, then we can get the shortest path length
        from other nodes to each labeled node
        """
        self.dis_matrix = -np.ones((self.adj.shape[0], self.nclass))
        for c, nodes in self.node2label.items():
            dis_matrix_ind = -np.ones((self.adj.shape[0], len(nodes)))
            for ii, node_st in enumerate(nodes):
                # do bfs
                visited = set([node_st])
                q = collections.deque([(x, 1) for x in self.graph.neighbors(node_st)])
                while q:
                    node_cur, depth = q.popleft()
                    if node_cur in visited:
                        continue
                    visited.add(node_cur)
                    dis_matrix_ind[node_cur][ii] = self.transform_depth(depth)
                    for node_next in self.graph.neighbors(node_cur):
                        q.append((node_next, depth+1))
            # dis_matrix_ind[dis_matrix_ind==-1] = dis_matrix_ind.max() + 2
            dis_matrix_ind[dis_matrix_ind==-1] = 0
            self.dis_matrix[:, c] = dis_matrix_ind.mean(1)


class ICAAgent:

    def __init__(self, adj, features, labels, idx_train, idx_test, args):
        """
        idx_train: labeled data
        idx_test: unlabeled data
        """

        self.args = args
        self.adj = adj.tocsr()
        if args.dataset != 'reddit':
            self.adj_two_hop = adj.dot(adj)
            self.adj_two_hop.setdiag(0)
            self.adj_two_hop.eliminate_zeros()

        # self.graph = nx.from_scipy_sparse_matrix(adj)
        self.pseudo_labels = np.zeros((adj.shape[0], labels.shape[1]))
        # load_data = os.path.exists(f'preds/ICA_probs_{args.dataset}_{args.seed}.npy')
        load_data = os.path.exists(f'preds/ICA_probs_{args.train_size}_{args.dataset}_{args.seed}.npy')
        print('if loading: ', load_data)
        if not load_data:
            st = time.time()
            if args.dataset != 'cora':
                features[features!=0] = 1
            classifier = 'sklearn.linear_model.LogisticRegression'
            aggregate = 'count' # choices=['count', 'prop'], help='Aggregation operator'

            graph, domain_labels = build_graph(adj, features, labels)
            y_true = [graph.node_list[t].label for t in idx_test]
            local_clf = LocalClassifier(classifier)
            agg = pick_aggregator(aggregate, domain_labels)
            relational_clf = RelationalClassifier(classifier, agg)
            ica = ICA(local_clf, relational_clf, bootstrap=True, max_iteration=10)

            ica.fit(graph, idx_train)
            conditional_node_to_label_map = create_map(graph, idx_train)

            eval_idx = np.setdiff1d(range(adj.shape[0]), idx_train)
            ica_predict, probs = ica.predict(graph, eval_idx, idx_test, conditional_node_to_label_map)
            ica_accuracy = accuracy_score(y_true, ica_predict)
            print('Acc: ' + str(ica_accuracy))
            print('optimization consumes %s s' % (time.time()-st))
            # self.ica_predict = np.array([int(x[1:]) for x in ica_predict])
            dict_pred = {x: int(y[1:]) for x, y in zip(idx_test, ica_predict)}
            dict_train = {x: labels.argmax(1)[x] for x in idx_train}
            dict_pred.update(dict_train)
            concated = sorted(dict_pred.items(), key=lambda x: x[0])

            self.probs = np.vstack((labels[idx_train], probs))
            self.concated = np.array([y for x, y in concated])

            np.save(f'preds/ICA_probs_{args.train_size}_{args.dataset}_{args.seed}.npy', self.probs)
            np.save(f'preds/ICA_preds_{args.train_size}_{args.dataset}_{args.seed}.npy', self.concated)

        else:
            print('loading probs/preds...')
            # self.probs = np.load(f'ICA_probs_{args.dataset}_{args.seed}.npy')
            # self.concated = np.load(f'ICA_preds_{args.dataset}_{args.seed}.npy')
            # self.probs = np.load(f'ICA_probs_{args.dataset}_10.npy')
            # self.concated = np.load(f'ICA_preds_{args.dataset}_10.npy')

            self.probs = np.load(f'preds/ICA_probs_{args.train_size}_{args.dataset}_{args.seed}.npy')
            self.concated = np.load(f'preds/ICA_preds_{args.train_size}_{args.dataset}_{args.seed}.npy')

            # self.probs = np.load(f'preds/{args.dataset}_{args.seed}_pred.npy')
            # self.concated = self.probs.argmax(1)
            # self.concated[idx_train] = labels.argmax(1)[idx_train]
            print('Acc: ', (self.concated == labels.argmax(1))[idx_test].sum()/len(idx_test))

    def get_label(self, concated=None, use_probs=False):
        '''
        Get neighbor label distribution
        '''
        if use_probs:
            pred = self.probs
        else:
            if concated is None:
                concated = self.concated
            pred = encode_onehot(concated)

        A = self.adj

        st = time.time()
        if self.args.dataset != 'reddit':
            B = self.adj_two_hop
            self.pseudo_labels = _get_subgraph_label(pred, self.pseudo_labels, A.indptr, A.indices, B.indptr, B.indices)
        else:
            self.pseudo_labels = _get_neighbor_label(pred, self.pseudo_labels, A.indptr, A.indices)
        print('building label consumes %s s' % (time.time()-st))
        return torch.FloatTensor(self.pseudo_labels)


class LPAgent:

    def __init__(self, adj, features, labels, idx_train, idx_test, args):
        """
        :param graph: Networkx Graph.
        """

        self.adj = adj.tocsr()
        self.adj_two_hop = adj.dot(adj)
        self.adj_two_hop.setdiag(0)
        self.adj_two_hop.eliminate_zeros()
        self.pseudo_labels = np.zeros((adj.shape[0], labels.shape[1]))

        # load_data = False
        load_data = os.path.exists(f'preds/LP_probs_{args.train_size}_{args.dataset}_{args.seed}.npy')
        if not load_data:
            self.graph = nx.from_scipy_sparse_matrix(adj)
            lp_predict, self.probs = self.hmn(labels.argmax(1), idx_train)
            # lp_predict = self.propagate(adj, labels.argmax(1), idx_train)
            lp_accuracy = accuracy_score(labels.argmax(1)[idx_test], lp_predict[idx_test])
            print('Acc: ' + str(lp_accuracy))
            dict_pred = {x: y for x, y in enumerate(lp_predict)}
            concated = sorted(dict_pred.items(), key=lambda x: x[0])
            self.concated = np.array([y for x, y in concated])

            # np.save(f'LP_probs_{args.dataset}_{args.seed}.npy', self.probs)
            # np.save(f'LP_preds_{args.dataset}_{args.seed}.npy', self.concated)
        else:
            # self.probs = np.load(f'LP_probs_{args.dataset}_{args.seed}.npy')
            # self.concated = np.load(f'LP_preds_{args.dataset}_{args.seed}.npy')
            # self.probs = np.load(f'LP_probs_{args.dataset}_10.npy')
            # self.concated = np.load(f'LP_preds_{args.dataset}_10.npy')
            self.probs = np.load(f'preds/LP_probs_{args.train_size}_{args.dataset}_{args.seed}.npy')
            self.concated = np.load(f'preds/LP_preds_{args.train_size}_{args.dataset}_{args.seed}.npy')

            print('Acc: ', (self.concated == labels.argmax(1))[idx_test].sum()/len(idx_test))

    def hmn(self, labels, idx_train):
        # from networkx.algorithms import node_classification
        import node_classification
        for id in idx_train:
            self.graph.nodes[id]['label'] = labels[id]
        preds, probs = node_classification.harmonic_function(self.graph)
        return np.array(preds), probs

    def propagate(self, adj, labels, idx_train):
        # row_sums = adj.sum(axis=1).A1
        # row_sum_diag_mat = np.diag(row_sums)
        # adj_rw = np.linalg.inv(row_sum_diag_mat).dot(adj)
        adj_rw = row_normalize(self.adj.asfptype())
        Y = np.zeros(labels.shape)
        for id in idx_train:
            Y[id] = labels[id]

        for i in range(0, 1000):
            Y = adj_rw.dot(Y)
            for id in idx_train:
                Y[id] = labels[id]  # Clamping

        return Y.round()

    def get_label(self, concated=None):
        '''
        Get neighbor label distribution
        '''
        if concated is None:
            concated = self.concated
        pred = encode_onehot(concated)
        A = self.adj
        B = self.adj_two_hop
        st = time.time()
        # self.pseudo_labels = _get_neighbor_label(pred, self.pseudo_labels, A.indptr, A.indices)
        self.pseudo_labels = _get_subgraph_label(pred, self.pseudo_labels, A.indptr, A.indices, B.indptr, B.indices)
        print('building label consumes %s s' % (time.time()-st))

        return torch.FloatTensor(self.pseudo_labels)


class CombinedAgent(ICAAgent):

    def __init__(self, adj, features, labels, idx_train, idx_test, args):
        """
        :param graph: Networkx Graph.
        """

        self.adj = adj
        self.args=args
        unlabeled = np.array([x for x in range(adj.shape[0]) if x not in idx_train])

        probs = np.zeros(labels.shape)
        # LP
        agent = LPAgent(adj, features, labels, idx_train, unlabeled, args)
        preds = agent.concated
        probs += agent.probs

        # #FeatLP

        # agent = FeatLPAgent(adj, features, labels, idx_train, unlabeled, args)
        # preds = agent.concated
        # probs += agent.probs
        # import ipdb
        # ipdb.set_trace()

        # ICA
        agent = ICAAgent(adj, features, labels, idx_train, unlabeled, args)
        preds_ica = agent.concated
        probs += agent.probs

        # # GCN?
        # probs_gcn = np.load(f'preds/{args.dataset}_{args.seed}_pred.npy')
        # probs += probs_gcn
        # accuracy = accuracy_score(probs_gcn.argmax(1)[unlabeled], labels.argmax(1)[unlabeled])
        # print('Acc: ' + str(accuracy))

        final_preds = probs.argmax(1)
        accuracy = accuracy_score(final_preds[unlabeled], labels.argmax(1)[unlabeled])
        print('Acc: ' + str(accuracy))

        dict_pred = {x: y for x, y in enumerate(final_preds)}
        concated = sorted(dict_pred.items(), key=lambda x: x[0])
        self.concated = np.array([y for x, y in concated])
        self.pseudo_labels = np.zeros((adj.shape[0], labels.shape[1]))

        self.adj = adj.tocsr()
        self.adj_two_hop = adj.dot(adj)
        self.adj_two_hop.setdiag(0)
        self.adj_two_hop.eliminate_zeros()


class AttrSim:

    def __init__(self, features, args, nclass=4):
        """
        :param graph: Networkx Graph.
        """
        self.features = features.cpu().numpy()
        # self.features[self.features!=0] = 1
        # self.graph = nx.from_scipy_sparse_matrix(adj)
        self.nclass = nclass
        self.args = args

    def get_label(self):
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.metrics import pairwise_distances
        from scipy.spatial.distance import jaccard
        # metric = jaccard
        metric = "cosine"
        # sims = pairwise_distances(self.features, self.features, metric=metric)
        sims = cosine_similarity(self.features)
        args = self.args

        k = 3
        print(f'loading {args.dataset}_{k}_attrsim_sampled_idx.npy')
        # if not os.path.exists(f'{args.dataset}_{k}_attrsim_sampled_idx.npy'):
        if True:
            indices_sorted = sims.argsort(1)
            idx = np.arange(k, sims.shape[0]-k)
            sampled = np.random.choice(idx, k, replace=False)
            selected = np.hstack((indices_sorted[:, :k],
                indices_sorted[:, -k-1:], indices_sorted[:, sampled]))

            from itertools import product
            selected_set = set()
            for i in range(len(sims)):
                for pair in product([i], selected[i]):
                    if pair[0] > pair[1]:
                        pair = (pair[1], pair[0])
                    if  pair[0] == pair[1]:
                        continue
                    selected_set.add(pair)

            sampled = np.array(list(selected_set)).transpose()
            np.save(f'{args.dataset}_{k}_attrsim_sampled_idx.npy', sampled)
        else:
            sampled = np.load(f'{args.dataset}_{k}_attrsim_sampled_idx.npy')
        print('number of sampled:', len(sampled[0]))
        self.node_pairs = (sampled[0], sampled[1])


        return torch.FloatTensor(sims[self.node_pairs])


@njit
def _get_neighbor_label(pred, pseudo_labels, iA, jA):
    '''
    Get neighbor label distribution
    '''
    for row in range(len(iA)-1):
        label_dist = pred[jA[iA[row]: iA[row+1]]].sum(0)
        pseudo_labels[row] = label_dist / label_dist.sum()
    return pseudo_labels

@njit
def _get_subgraph_label(pred, pseudo_labels, iA, jA, iB, jB):
    '''
    Get neighbor label distribution
    '''
    for row in range(len(iA)-1):
        label_dist = pred[jA[iA[row]: iA[row+1]]].sum(0)
        label_dist += pred[jB[iB[row]: iB[row+1]]].sum(0)
        pseudo_labels[row] = label_dist / label_dist.sum()
    return pseudo_labels



