import os
from collections import defaultdict
import numpy as np
import torch
import gzip
import pickle
from torch_geometric import datasets


def pickleLoad(path):
    """
    Load pickle data for MMORPG
    :param path: path to load the MMORPG data
    :return:
    """
    data = {}
    if os.path.exists(path):
       with gzip.open(path, 'rb') as f:
           data = pickle.load(f)
    else:
        print("No data")
    return data


def preprocess_edges(edges):
    """
    Preprocess edges to make sure the following:
    1) No self-loops.
    2) Each pair (a, b) and (b, a) exists exactly once.
    """
    m = defaultdict(lambda: set())
    for src, dst in edges.t():
        src = src.item()
        dst = dst.item()
        if src != dst:
            m[src].add(dst)
            m[dst].add(src)

    edges = []
    for src in sorted(m):
        for dst in sorted(m[src]):
            edges.append((src, dst))
    return np.array(edges, dtype=np.int64).transpose()


def to_pu_setting(labels):
    """
    Make PU labels by selecting the most frequent one as positive.
    """
    count = np.bincount(labels)
    positive_nodes = labels == count.argmax()
    pu_labels = np.zeros_like(labels)
    pu_labels[positive_nodes] = 1
    return pu_labels


def split_nodes(labels, trn_ratio, true_negative=None, seed=0):
    """
    Split nodes into training, validation, and test.
    """
    if true_negative is None:
        true_negative = []
    state = np.random.RandomState(seed)
    all_nodes = np.arange(labels.shape[0])
    pos_nodes = all_nodes[labels == 1]
    neg_nodes = all_nodes[labels == 0]

    n_pos_nodes = pos_nodes.shape[0]
    n_trn_nodes = int(n_pos_nodes * trn_ratio)
    n_test_pos_nodes = int((n_pos_nodes - n_trn_nodes))

    trn_nodes = state.choice(pos_nodes, size=n_trn_nodes, replace=False)

    test_pos_candidates = set(pos_nodes).difference(set(trn_nodes))
    test_nodes = state.choice(list(test_pos_candidates), size=n_test_pos_nodes, replace=False)
    if len(true_negative) > 0:
        test_nodes = np.concatenate([test_nodes, true_negative])
    else:
        test_nodes = np.concatenate([test_nodes, neg_nodes])

    return trn_nodes, test_nodes


def read_data(dataset, trn_ratio):
    """
    read & load the data to use it with the model
    :param dataset: name of the data to use
    :param trn_ratio: train ratio to decide how many positive will we use.
    :return: node_x: features of node
             node_y: true labels of node
             edges: edges
             trn_nodes: indexes of training nodes
             test_nodes: indexes of test nodes
    """
    root = '../data'
    root_cached = os.path.join(root, 'cached', dataset)
    root_raw = os.path.join(root, 'raw', dataset)
    if not os.path.exists(root_cached):
        if dataset == 'cora':
            data = datasets.Planetoid(root, 'Cora')
        elif dataset == 'citeseer':
            data = datasets.Planetoid(root, 'CiteSeer')
        elif dataset == 'cora-ml':
            data = datasets.CitationFull(root, 'Cora_ML')
        elif dataset == 'wikics':
            data = datasets.wikics.WikiCS(os.path.join(root, 'wikics'))
        elif dataset == "mmorpg":
            print("Creating MMORPG Data...")
        else:
            raise ValueError(dataset)

        if dataset == 'mmorpg':
            node_x = pickleLoad(root_raw+"/node")
            node_y = pickleLoad(root_raw+"/label")
            edges = np.array(pickleLoad(root_raw+"/edge_index"), dtype=np.int64).T
            true_negative = pickleLoad(root_raw+"/true_negative")
            os.makedirs(root_cached, exist_ok=True)
            np.save(os.path.join(root_cached, 'true_negative'), true_negative)

        else:
            node_x = data.data.x
            node_x[node_x.sum(dim=1) == 0] = 1
            node_x = node_x / node_x.sum(dim=1, keepdim=True)
            node_y = to_pu_setting(data.data.y)
            edges = preprocess_edges(data.data.edge_index)

        os.makedirs(root_cached, exist_ok=True)
        np.save(os.path.join(root_cached, 'x'), node_x)
        np.save(os.path.join(root_cached, 'y'), node_y)
        np.save(os.path.join(root_cached, 'edges'), edges)

    node_x = torch.from_numpy(np.array(np.load(os.path.join(root_cached, 'x.npy'), allow_pickle=True), dtype=np.float32))
    node_y = torch.from_numpy(np.array(np.load(os.path.join(root_cached, 'y.npy'), allow_pickle=True), dtype=np.int64))
    edges = torch.from_numpy(np.load(os.path.join(root_cached, 'edges.npy'), allow_pickle=True))

    if dataset == 'mmorpg':
        true_negative = torch.from_numpy(np.load(os.path.join(root_cached, 'true_negative.npy'), allow_pickle=True))
    else:
        true_negative = None

    trn_nodes, test_nodes = split_nodes(node_y, trn_ratio, true_negative)

    return node_x, node_y, edges, trn_nodes, test_nodes
