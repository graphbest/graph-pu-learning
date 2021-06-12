from torch import nn, sparse
import torch
import numpy as np


class SigmoidLoss(nn.Module):
    """
    Sigmoid loss for the non-negative risk estimator.
    """

    def __init__(self, reduction=True):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.reduction = reduction

    def forward(self, predictions, labels):
        labels = labels * 2 - 1  # changes {+1, 0} labels to {+1, -1} labels.
        loss = self.sigmoid(-predictions * labels)
        if self.reduction:
            loss = loss.mean()
        return loss

class InferenceModel(nn.Module):
    """
    Inference model to calculate belief propagation for each nodes
    """
    def __init__(self, edges, potential=0.95, threshold=1e-6, max_iters=100):
        super().__init__()

        if isinstance(edges, np.ndarray):
            values = torch.ones(edges.shape[0])
            edges = sparse.FloatTensor(torch.from_numpy(edges).t(), values)
        self.threshold = threshold
        self.max_iters = max_iters
        self.softmax = nn.Softmax(dim=1)

        indices = edges.coalesce().indices()
        self.src_nodes = nn.Parameter(indices[0, :], requires_grad=False)
        self.dst_nodes = nn.Parameter(indices[1, :], requires_grad=False)
        self.num_nodes = edges.size(0)

        # noinspection PyProtectedMember
        self.num_edges = edges._nnz() // 2
        self.rev_edges = nn.Parameter(self.set_rev_edges(edges), requires_grad=False)
        self.potential = nn.Parameter(torch.full([2, 2], fill_value=(1 - potential) / 2), requires_grad=False)
        self.potential[0, 0] = potential / 2
        self.potential[1, 1] = potential / 2

    def set_rev_edges(self, edges):
        degrees = sparse.mm(edges, torch.ones([self.num_nodes, 1])).view(-1).int()
        zero = torch.zeros(1, dtype=torch.int64)
        indices = torch.cat([zero, degrees.cumsum(dim=0)[:-1]])
        counts = torch.zeros(self.num_nodes, dtype=torch.int64)
        rev_edges = torch.zeros(2 * self.num_edges, dtype=torch.int64)
        edge_idx = 0
        for dst, degree in enumerate(degrees):
            for _ in range(degree):
                src = self.dst_nodes[edge_idx]
                rev_edges[indices[src] + counts[src]] = edge_idx
                edge_idx += 1
                counts[src] += 1
        return rev_edges

    def update_messages(self, messages, beliefs):
        new_beliefs = beliefs[self.src_nodes]
        rev_messages = messages[self.rev_edges]
        new_msgs = torch.mm(new_beliefs / rev_messages, self.potential)
        return new_msgs / new_msgs.sum(dim=1, keepdim=True)

    def compute_beliefs(self, priors, messages):
        beliefs = priors.log()
        beliefs.index_add_(0, self.dst_nodes, messages.log())
        return self.softmax(beliefs)

    def forward(self, priors):
        beliefs = priors
        messages = torch.full([self.num_edges * 2, 2], fill_value=0.5, device=priors.device)
        for _ in range(self.max_iters):
            old_beliefs = beliefs
            messages = self.update_messages(messages, beliefs)
            beliefs = self.compute_beliefs(priors, messages)
            diff = (beliefs - old_beliefs).abs().max()
            if diff < self.threshold:
                break
        return beliefs


class BeliefRiskEstimator(nn.Module):
    """
    The novel belief risk estimator method which used in our GRAB method to calculate risk of loss.
    """
    def __init__(self, edges, priors, potential=0.9, recompute=False, labels=None):
        super().__init__()
        if isinstance(priors, float):
            self.pi = priors
            assert labels is not None
        if recompute:
            priors = self.to_priors(labels, priors)
            model = InferenceModel(edges,potential)
            self.marginals = nn.Parameter(model(priors), requires_grad=False)
        else:
            priors = self.to_initial_priors(labels)
            self.marginals = priors

        self.loss = SigmoidLoss(reduction=False)

    @staticmethod
    def to_priors(labels, prior):
        num_nodes = labels.size(0)
        priors = torch.zeros(num_nodes, 2, device=labels.device)
        priors[labels == 1, 0] = 0
        priors[labels == 1, 1] = 1
        priors[labels == 2, 0] = prior
        priors[labels == 2, 1] = 1 - prior
        priors[labels == 0, 0] = 1 - prior
        priors[labels == 0, 1] = prior
        return priors

    @staticmethod
    def to_initial_priors(labels):
        num_nodes = labels.size(0)
        priors = torch.zeros(num_nodes, 2, device=labels.device)
        priors[labels == 1, 0] = 0
        priors[labels == 1, 1] = 1
        priors[labels == 0, 0] = 1
        priors[labels == 0, 1] = 0
        return priors

    def forward(self, predictions, labels):
        all_nodes = torch.arange(predictions.size(0), device=predictions.device)
        pos_nodes = all_nodes[labels == 1]
        unl_nodes = all_nodes[labels == 0]
        r_hat_plus_p = self.loss(predictions[pos_nodes], 1).mean()
        r_hat_plus_u = (self.loss(predictions[unl_nodes], 1) * self.marginals[unl_nodes, 1].to(predictions.device)).mean()
        r_hat_minus_u = (self.loss(predictions[unl_nodes], 0) * self.marginals[unl_nodes, 0].to(predictions.device)).mean()
        return r_hat_plus_p + r_hat_plus_u + r_hat_minus_u
