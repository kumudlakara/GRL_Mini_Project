# This implementation is based on https://github.com/weihua916/powerful-gnns, https://github.com/chrsmrrs/k-gnn/tree/master/examples and https://github.com/KarolisMart/DropGNN
# Datasets are implemented based on the description in the corresonding papers
import argparse
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv
torch.set_printoptions(profile="full")
from syn_datasets import SkipCircles, Triangles, LCC, LimitsOne, LimitsTwo, FourCycles
from gnn_model import GIN
from dropgnn_model import DropGIN


def train(epoch, model, device, use_aux_loss, loader, optimizer):
        model.train()
        loss_all = 0
        n = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            logs, aux_logs = model(data)
            loss = F.nll_loss(logs, data.y)
            n += len(data.y)
            if use_aux_loss:
                aux_loss = F.nll_loss(aux_logs.view(-1, aux_logs.size(-1)), data.y.unsqueeze(0).expand(aux_logs.size(0),-1).clone().view(-1))
                loss = 0.75*loss + 0.25*aux_loss
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            optimizer.step()
        return loss_all / n

def test(loader, model, device):
        model.eval()
        n = 0
        with torch.no_grad():
            correct = 0
            for data in loader:
                data = data.to(device)
                logs, aux_logs = model(data)
                pred = logs.max(1)[1]
                n += len(pred)
                correct += pred.eq(data.y).sum().item()
        return correct / n

def train_and_test(model, num_runs, dataset, device, use_aux_loss, multiple_tests=False, test_over_runs=None):
        train_accs = []
        test_accs = []
        #nonlocal num_runs # access global num_runs variable inside this function
        print(model.__class__.__name__)
        for seed in range(10):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model.reset_parameters()
            lr = 0.01
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            test_dataset = dataset.makedata()
            train_dataset = dataset.makedata()

            test_loader = DataLoader(test_dataset, batch_size=len(train_dataset))
            train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

            print('---------------- Seed {} ----------------'.format(seed))
            for epoch in range(1, 1001):
                if args.verbose:
                    start = time.time()
                train_loss = train(epoch, model, device, use_aux_loss, train_loader, optimizer)
                if args.verbose:
                    print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, Time: {:7f}'.format(epoch, lr, train_loss, time.time() - start), flush=True)
            train_acc = test(train_loader,model, device)
            train_accs.append(train_acc)
            if not test_over_runs is None:
                if multiple_tests:
                    for i in range(10):
                        old_num_runs = num_runs
                        for r in test_over_runs:
                            num_runs = r
                            test_acc = test(test_loader, model, device)
                            test_accs.append(test_acc)
                        num_runs = old_num_runs
                else:
                    old_num_runs = num_runs
                    for r in test_over_runs:
                        num_runs = r
                        test_acc = test(test_loader, model, device)
                        test_accs.append(test_acc)
                    num_runs = old_num_runs
            elif multiple_tests:
                for i in range(10):
                    test_acc = test(test_loader, model, device)
                    test_accs.append(test_acc)
                test_acc =  torch.tensor(test_accs[-10:]).mean().item()
            else:
                test_acc = test(test_loader, model, device)
                test_accs.append(test_acc)
            print('Train Acc: {:.7f}, Test Acc: {:7f}'.format(train_acc, test_acc), flush=True)            
        train_acc = torch.tensor(train_accs)
        test_acc = torch.tensor(test_accs)
        if not test_over_runs is None:
            test_acc = test_acc.view(-1, len(test_over_runs))
        print('---------------- Final Result ----------------')
        print('Train Mean: {:7f}, Train Std: {:7f}, Test Mean: {}, Test Std: {}'.format(train_acc.mean(), train_acc.std(), test_acc.mean(dim=0), test_acc.std(dim=0)), flush=True)
        return test_acc.mean(dim=0), test_acc.std(dim=0)


def main(args, cluster=None):
    print(args, flush=True)

    if args.dataset == "skipcircles":
        dataset = SkipCircles()
    elif args.dataset == "triangles":
        dataset = Triangles()
    elif args.dataset == "lcc":
        dataset = LCC()
    elif args.dataset == "limitsone":
        dataset = LimitsOne()
    elif args.dataset == "limitstwo":
        dataset = LimitsTwo()
    elif args.dataset == "fourcycles":
        dataset = FourCycles()

    print(dataset.__class__.__name__)

    # Set the sampling probability and number of runs/samples for the DropGIN
    n = dataset.num_nodes
    print(f'Number of nodes: {n}')
    gamma = n
    p_opt = 2 * 1 /(1+gamma)
    if args.prob >= 0:
        p = args.prob
    else:
        p = p_opt
    if args.num_runs > 0:
        num_runs = args.num_runs
    else:
        num_runs = gamma
    print(f'Number of runs: {num_runs}')
    print(f'Sampling probability: {p}')

    degs = []
    for g in dataset.makedata():
        deg = degree(g.edge_index[0], g.num_nodes, dtype=torch.long)
        degs.append(deg.max())
    print(f'Mean Degree: {torch.stack(degs).float().mean()}')
    print(f'Max Degree: {torch.stack(degs).max()}')
    print(f'Min Degree: {torch.stack(degs).min()}')
    print(f'Number of graphs: {len(dataset.makedata())}')
    
    num_features = dataset.num_features
    if args.augmentation == 'random' or args.augmentation == 'prob_rni':
        num_features += 1

    use_aux_loss = args.use_aux_loss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if args.augmentation == 'dropout' or args.augmentation == 'rrni':
        model = DropGIN(dataset, args.num_layers, num_features, use_aux_loss, num_runs, p, args.augmentation).to(device)
    else:
        model = GIN(dataset, num_features, args.num_layers, args.augmentation, p).to(device)
        use_aux_loss = False

    if args.prob_experiment:
        print('Dropout probability Experiment')
        probs = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16]
        means = []
        stds = []
        for prob in probs:
            print(f'Dropout probability {prob}:')
            p = prob
            mean, std = train_and_test(model, num_runs, dataset, device, use_aux_loss, multiple_tests=True)
            means.append(mean.item())
            stds.append(std.item())
        probs = np.array(probs)
        means = np.array(means)
        stds = np.array(stds)
        lower = means - stds
        lower = [i if i > 0 else 0 for i in lower]
        upper = means + stds
        upper = [i if i <= 1 else 1 for i in upper]
        plt.plot(probs, means)
        plt.fill_between(probs, lower, upper, alpha=0.3)
        plt.xlabel("Dropout Probability")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.ylim(bottom=0.4)
        plt.vlines(p_opt, 0, 2, colors="k")
        file_name = "experiment_p_dropgnn_{}.pdf".format(args.dataset)
        plt.savefig(file_name)
    elif args.rrni_experiment:
        random_vals = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
        means = []
        stds = []
        for val in random_vals:
            print(f'Random value {val}:')
            model.random_value = val
            mean, std = train_and_test(model, num_runs, dataset, device, use_aux_loss, multiple_tests=True)
            means.append(mean.item())
            stds.append(std.item())
        random_vals = np.array(random_vals)
        means = np.array(means)
        stds = np.array(stds)
        lower = means - stds
        lower = [i if i > 0 else 0 for i in lower]
        upper = means + stds
        upper = [i if i <= 1 else 1 for i in upper]
        plt.plot(random_vals, means)
        plt.fill_between(random_vals, lower, upper, alpha=0.3)
        plt.xlabel("Random Value")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.ylim(bottom=0.4)
        file_name = "experiment_rrni_{}.pdf".format(args.dataset)
        plt.savefig(file_name)
    elif args.rni_experiment:
        print('RNI Experiment')
        probs = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16]
        means = []
        stds = []
        for prob in probs:
            print(f'Randomization probability {prob}:')
            p = prob
            mean, std = train_and_test(model, num_runs, dataset, device, use_aux_loss, multiple_tests=True)
            means.append(mean.item())
            stds.append(std.item())
        probs = np.array(probs)
        means = np.array(means)
        stds = np.array(stds)
        lower = means - stds
        lower = [i if i > 0 else 0 for i in lower]
        upper = means + stds
        upper = [i if i <= 1 else 1 for i in upper]
        plt.plot(probs, means)
        plt.fill_between(probs, lower, upper, alpha=0.3)
        plt.xlabel("Dropout Probability")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.ylim(bottom=0.4)
        plt.vlines(p_opt, 0, 2, colors="k")
        file_name = "experiment_p_rni_{}.pdf".format(args.dataset)
        plt.savefig(file_name)
    else:
        train_and_test(model, num_runs, dataset, device, use_aux_loss,)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--augmentation', type=str, default='none', help="Options are ['none', 'random', 'dropout', 'prob_rni', 'rrni']")
    parser.add_argument('--prob', type=int, default=-1)
    parser.add_argument('--num_runs', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=4) # 9 layers to be used for skipcircles dataset
    parser.add_argument('--use_aux_loss', action='store_true', default=False)
    
    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--prob_experiment', action='store_true', default=False, help="Run probability experiments")
    parser.add_argument('--rrni_experiment', action='store_true', default=False, help="Run rRNI experiment with different random values")
    parser.add_argument('--rni_experiment', action='store_true', default=False, help="run rni experiment")
    parser.add_argument('--dataset', type=str, default='limitsone', help="Options are ['skipcircles', 'triangles', 'lcc', 'limitsone', 'limitstwo', 'fourcycles']")
    args = parser.parse_args()

    main(args)

    print('Finished', flush=True)
