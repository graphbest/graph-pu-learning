import argparse
import io
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
import torch
from torch import optim
import data
import models


def to_device(gpu):
    """
    make torch to use GPU
    :param gpu: gpu to use
    :return:
    """
    if gpu is not None and torch.cuda.is_available():
        return torch.device('cuda:{}'.format(gpu))
    else:
        return torch.device('cpu')


def parse_args():
    """
    parser arguments to run program in cmd
    :return:
    """
    parser = argparse.ArgumentParser()

    # Pre-sets before start the program
    parser.add_argument('--data', type=str, default='cora')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    # Hyperparameters for training
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--trn-ratio', type=float, default=0.5)

    # Hyperparameters for models
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--units', type=int, default=16)
    parser.add_argument('--mu-iters', type=int, default=10)
    parser.add_argument('--potential', type=float, default=0.9)

    return parser.parse_args()


def compute_prior(trn_labels):
    """
    compute the class prior, which indicates the number of positive nodes among the all nodes.
    """
    return (trn_labels.sum().float() / len(trn_labels)).item()


def train_model(model, features, edges, labels, test_nodes, loss_func, optimizer,
                trn_labels, epochs, patience):
    """
    train the model.
    :param model: model to use.
    :param features: features of data to use
    :param edges: edge information of data
    :param labels: true labels of data (transductive)
    :param test_nodes: index list of test nodes.
    :param loss_func: loss function for model
    :param optimizer: optimizer for model
    :param trn_labels: train labels of data which is actually used to train model
    :param epochs: number of epochs to run
    :param patience: number of patience
    :return:
    """
    logs = []
    saved_model, best_epoch, best_f1 = io.BytesIO(), -1, -1
    best_loss = np.inf
    for epoch in range(epochs + 1):
        model.train()
        out = model(features, edges)
        out = out[:len(labels)]
        loss = loss_func(out, trn_labels)

        if epoch > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_f1, test_auc = evaluate_model(
            model, features, edges, labels, test_nodes)

        logs.append((epoch, loss.item(), test_f1, test_auc))
        if loss.item() < best_loss:
            best_epoch = epoch
            best_loss = loss.item()
            saved_model.seek(0)
            torch.save(model.state_dict(), saved_model)

        if patience > 0 and epoch >= best_epoch + patience:
            break

    saved_model.seek(0)
    model.load_state_dict(torch.load(saved_model))

    columns = ['epoch', 'trn_loss', 'test_f1', 'test_acc']
    return best_epoch, pd.DataFrame(logs, columns=columns), best_loss, model


def evaluate_model(model, features, edges, labels, test_nodes):
    """
    evaluate the model with true label to check the final performances
    :param model: trained model
    :param features: features of node
    :param edges: edges of data
    :param labels: true labels of node
    :param test_nodes: indexes of test nodes
    :return:
    """
    model.eval()
    with torch.no_grad():
        out = torch.sigmoid(model(features, edges)).cpu()
        out_labels = labels.clone()
        out_labels[out > 0.5] = 1
        out_labels[out < 0.5] = 0
    test_f1 = f1_score(labels[test_nodes], out_labels[test_nodes])
    test_acc = accuracy_score(labels[test_nodes], out_labels[test_nodes])
    return test_f1, test_acc


def get_expectation(trn_nodes, trn_labels, predictions):
    """
    compute expected posterior & modify trn labels to apply expected posterior
    :param trn_nodes: indexes of training nodes
    :param trn_labels: Training labels of nodes
    :param predictions: result from the model
    :return:
    """
    expected_trn_labels = trn_labels.clone()
    expected_trn_labels[predictions > 0.50] = 1
    expected_trn_labels[predictions < 0.50] = 0
    prior_labels = expected_trn_labels.cpu().numpy()
    prior_labels = torch.from_numpy(np.delete(prior_labels, trn_nodes))
    prior = compute_prior(prior_labels)
    expected_trn_labels = trn_labels.clone()
    expected_trn_labels[predictions > 0.50] = 2
    expected_trn_labels[trn_nodes] = 1
    return prior, expected_trn_labels


def main():
    # initial setting for torch
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = to_device(args.gpu)

    # Call the designated data
    features, labels, edges, trn_nodes, test_nodes = data.read_data(args.data, args.trn_ratio)
    num_nodes = features.size(0)
    num_features = features.size(1)
    trn_labels = torch.zeros(num_nodes, dtype=torch.float)
    trn_labels[trn_nodes] = 1

    # padding for NC sparse version graphs
    padding = torch.zeros(len(trn_labels)-len(labels), dtype=int)
    labels = torch.cat((labels, padding), dim=0)

    # Initial all variables related to model
    model = models.GCN(num_features, args.units, args.layers).to(device)
    optimizer = optim.Adam(model.parameters())
    prior = compute_prior(trn_labels)
    loss_func = models.BeliefRiskEstimator(edges.t().numpy(),
                                           prior, args.potential, False, trn_labels).to(device)
    features = features.to(device)
    edges = edges.to(device)
    trn_labels = trn_labels.to(device)

    # Train the model (Initial step GCN)
    best_epoch, logs, best_loss, best_model = train_model(
        model, features, edges, labels, test_nodes, loss_func, optimizer, trn_labels,
        args.epochs, args.patience)

    the_best_loss = best_loss
    old_logs = logs
    old_epoch = best_epoch
    old_model = best_model
    print("Loss Before MU step Starts: {:.4}".format(best_loss))

    rst = logs.loc[best_epoch:best_epoch]
    predictions = torch.sigmoid(best_model(features, edges))

    expected_PN, expected_labels = get_expectation(trn_nodes, trn_labels, predictions)
    RPN = labels.cpu().numpy()
    RPN = torch.from_numpy(np.delete(RPN, trn_nodes))
    real_PN = compute_prior(RPN)
    rst = rst.reset_index().join(pd.DataFrame({'expected_PN': [expected_PN], 'real_PN': [real_PN]}))

    # MU iteration Starts
    for i in range(args.mu_iters):
        with torch.no_grad():
            predictions = torch.sigmoid(best_model(features, edges))
            posterior, expected_labels = get_expectation(trn_nodes, trn_labels, predictions)

        model = models.GCN(num_features, args.units, args.layers).to(device)
        optimizer = optim.Adam(model.parameters())
        loss_func = models.BeliefRiskEstimator(edges.cpu().t().numpy(),
                                               posterior, args.potential, True, expected_labels.cpu()).to(device)

        best_epoch, logs, best_loss, best_model = train_model(
            model, features, edges, labels, test_nodes, loss_func, optimizer, trn_labels,
            args.epochs, args.patience)
        print("Loss at MU step ", i+1, ": {:.4}".format(best_loss))

        predictions = torch.sigmoid(best_model(features, edges))

        expected_PN, expected_labels = get_expectation(trn_nodes, trn_labels, predictions)
        RPN = labels.cpu().numpy()
        RPN = torch.from_numpy(np.delete(RPN, trn_nodes))
        real_PN = compute_prior(RPN)
        rst = rst.append(logs.loc[best_epoch:best_epoch].reset_index().join(pd.DataFrame(
            {'expected_PN': [expected_PN], 'real_PN': [real_PN]})), ignore_index=True)

        if best_loss > the_best_loss:
            logs = old_logs
            best_epoch = old_epoch
            best_model = old_model
            break
        else:
            the_best_loss = best_loss
            old_logs = logs
            old_epoch = best_epoch
            old_model = best_model

    with torch.no_grad():
        predictions = torch.sigmoid(best_model(features, edges))
        expected_PN, expected_labels = get_expectation(trn_nodes, trn_labels, predictions)
        RPN = labels.cpu().numpy()
        RPN = torch.from_numpy(np.delete(RPN, trn_nodes))

    real_PN = compute_prior(RPN)
    result = logs.loc[best_epoch:best_epoch]
    result = result.reset_index().drop(result.columns[[0]], axis=1).join(pd.DataFrame({'expected_PN': [expected_PN],
                                                                                       'real_PN': [real_PN]}))
    print(rst, '\n')
    print(result)

if __name__ == '__main__':
    main()

