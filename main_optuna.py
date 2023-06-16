import torch
import random
import argparse
import time
from dataset import get_dataset

from util import Logger, str2bool, spectral
from get_model import get_model
from train_eval import train, test
from train_eval import train_altopt, test_altopt, train_cs, test_cs, test1, train_appnp, test_appnp
from model import CorrectAndSmooth

import optuna
from myutil import sort_trials

def parse_args():
    parser = argparse.ArgumentParser(description='ALTOPT')
    parser.add_argument('--seed', type=int, default=12321312)
    parser.add_argument('--test', type=str2bool, default=False)
    parser.add_argument('--log_steps', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--gnnepoch', type=int, default=100)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--normalize_features', type=str2bool, default=True, help="whether to normalize node feature")
    parser.add_argument('--random_splits', type=int, default=0, help='default use fix split')

    parser.add_argument('--model', type=str, default='ALTOPT')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--Fwd', type=float, default=None)
    parser.add_argument('--earlystop', type=int, default=10)
    parser.add_argument('--lr', type=float, default=None)

    parser.add_argument('--prop', type=str, default='EMP') # useless
    parser.add_argument('--bn', type=str2bool, default=False) # number of propagation
    parser.add_argument('--tailln', type=str2bool, default=False) # number of propagation
    parser.add_argument('--K0', type=int, default=None) # number of propagation
    parser.add_argument('--K', type=int, default=None) # number of propagation
    parser.add_argument('--gamma', type=float, default=None) # used in EMP prop
    parser.add_argument('--lambda1', type=float, default=None)
    parser.add_argument('--lambda2', type=float, default=None)
    parser.add_argument('--L21', type=str2bool, default=True) # useless
    parser.add_argument('--alpha', type=float, default=None) # PPR alpha
    
    parser.add_argument('--defense', type=str, default=None) # no use
    parser.add_argument('--ptb_rate', type=float, default=0) # no use
    parser.add_argument('--sort_key', type=str, default='K')
    parser.add_argument('--debug', type=str2bool, default=False) # no use

    parser.add_argument('--softmaxF', type=str2bool, default=True)
    parser.add_argument('--useGCN', type=str2bool, default=True)
    parser.add_argument("--onlyy", type=str2bool, default=False)
    parser.add_argument("--usecg", type=str2bool, default=True)
    parser.add_argument("--weightedloss", type=str2bool, default=True)

    parser.add_argument("--temperature", type=float, default=0.2)
    
    parser.add_argument('--loss', type=str, default=None, choices=["CE", "MSE"])
    parser.add_argument('--LP', type=str2bool, default=False, help='Label propagation') #only in EMP
    parser.add_argument('--loop', type=int, default=None, help='Iteration number of MLP each epoch')
    parser.add_argument('--fix_num', type=int, default=0, help='number of train sample each class')
    parser.add_argument('--proportion', type=float, default=0, help='proportion of train sample each class')
    parser.add_argument('--has_weight', type=str2bool, default=True) # no use
    parser.add_argument('--noise', type=float, default=0, help='label noise ratio')
    parser.add_argument('--num_correct_layer', type=int, default=None)
    parser.add_argument('--correct_alpha', type=float, default=None)
    parser.add_argument('--num_smooth_layer', type=int, default=None)
    parser.add_argument('--smooth_alpha', type=float, default=None)
    parser.add_argument('--spectral', type=str2bool, default=False) # spectral embedding
    parser.add_argument('--pro_alpha', type=float, default=None)
    parser.add_argument('--const_split', type=str2bool, default=False)

    args = parser.parse_args()
    args.ogb = True if 'ogb' in args.dataset.lower() else False
    if args.K0 is None:
        args.K0 = args.K
    return args

def objective(trial=None):
    args = parse_args()
    if trial is not None:
        args = set_up_trial(trial, args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    if args.random_splits > 0:
        random_split_num = args.random_splits
        print(f'random split {random_split_num} times and each for {args.runs} runs')
    else:
        random_split_num = 1
        print(f'fix split and run {args.runs} times')

    logger = Logger(args.runs * random_split_num)

    total_start = time.perf_counter()

    ## data split
    for split in range(random_split_num):
        dataset, data, split_idx = get_dataset(args, split, defense=args.defense)
        data.psuedo_indices = None
        if args.spectral:
            data.x = torch.cat([data.x, spectral(data)], dim=-1)
            all_features = data.num_features
        else:
            all_features = data.num_features
        # print('feature', data.num_features)
        args.num_class = data.y.max()+1
        train_idx = split_idx['train']
        print("Data:", data)
        ## add noise
        mask = data.train_mask
        num_train = mask.sum()
        print('num_train', num_train)
        num_noise = int(args.noise * num_train)
        print('num_noise', num_noise)
        y = data.y.clone()
        if num_noise != 0:
            indices = torch.randperm(num_train)[:num_noise]
            rand_idx = train_idx[indices]
            data.y[rand_idx] = torch.randint(args.num_class, (num_noise,))
        print('noise:', (data.y != y).sum())

        data = data.to(device)
        if not isinstance(data.adj_t, torch.Tensor):
            data.adj_t = data.adj_t.to_symmetric()

        if args.ogb:
            args.num_layers = 3
            args.weight_decay = 0
            args.hidden_channels = 256
        start = time.time()
        model = get_model(args, dataset, all_features)
        print(model)
        if args.model == 'LP':
            result = test(model, data, split_idx, args=args)
            logger.add_result(split, result)
            continue

        model.reset_parameters()
        if args.model in ['IAPPNP', 'ORTGNN', 'ALTOPT', "AGD", "EXACT", 'APPNP', 'MLP', 'CS']:
            model.propagate(data, args.K0)
            print('propagate done')

        for run in range(args.runs):
            data.pseudo_mask = data.train_mask.clone()
            data.pseudo_label = data.y.clone()

            data.total_weight = torch.zeros(data.y.shape[0]).cuda()
            data.f = None
            runs_overall = split * args.runs + run
            model.reset_parameters()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            t_start = time.perf_counter()
            best_acc = 0
            y_soft = None
            nic_step = 0
            args.current_epoch = 0
            if args.model in ['ALTOPT', "EXACT", "AGD"]:
                for i in range(args.gnnepoch):
                    loss = train(model, data, train_idx, optimizer, args=args) #
                # train_altopt(model, data, train_idx, optimizer, args=args) # 
                # result = test(model, data, split_idx, args=args)
                # print('vanilla GNN test_result', result)
                args.current_epoch = 1
            for epoch in range(1, 1 + args.epochs):
                args.current_epoch = epoch
                if args.model in ['ALTOPT', "EXACT", "AGD"]:
                    loss = 0
                    model.propagate_update(data, K=args.K)
                    for ii in range(args.loop):
                        loss = train_altopt(model, data, train_idx, optimizer, args=args)
                    result = test_altopt(model, data, split_idx, args=args)
                elif args.model == 'CS':
                    loss = train_cs(model, data, train_idx, optimizer, args=args)
                    train_acc, val_acc, test_acc, out = test_cs(model, data, split_idx, args=args)
                    if val_acc > best_acc:
                        best_acc = val_acc
                        y_soft = out.softmax(dim=-1)
                    result = [train_acc, val_acc, test_acc]
                elif args.model == 'APPNP':
                    loss = train_appnp(model, data, train_idx, optimizer, args=args)
                    result = test_appnp(model, data, split_idx, args=args)
                else:
                    loss = train(model, data, train_idx, optimizer, args=args)
                    result = test(model, data, split_idx, args=args)

                if args.model != 'CS':
                    logger.add_result(runs_overall, result)
                if args.log_steps > 0:
                    if epoch % args.log_steps == 0:
                        train_acc, valid_acc, test_acc = result
                        print(f'Split: {split + 1:02d}, '
                              f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_acc:.2f}%, '
                              f'Valid: {100 * valid_acc:.2f}% '
                              f'Test: {100 * test_acc:.2f}%')
                train_acc, valid_acc, test_acc = result
                nic_step += 1
                if best_acc <= valid_acc:
                    nic_step = 0
                    best_acc = valid_acc
                # print(nic_step, best_acc)
                if nic_step > args.earlystop:
                    break
            if args.model == 'CS':
                print('best_acc', best_acc)
                # adj_t = data.adj_t.to(device)
                # deg = adj_t.sum(dim=1).to(torch.float)
                # deg_inv_sqrt = deg.pow_(-0.5)
                # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                # DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
                # DA = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj_t
                # torch.save(y_soft, 'y_soft2.th')

                if y_soft is None:
                    y_soft = torch.load('y_soft2.th')
                test1(model, data, y_soft, split_idx, args=args)
                CS = CorrectAndSmooth(args)
                print('correct and smooth')
                y_soft = CS.correct(data=data, mlp=y_soft, edge_weight=None)
                y_soft = CS.smooth(data=data, y_soft=y_soft, edge_weight=None)
                print('Done')
                train_acc, val_acc, test_acc, out = test_cs(model, data, split_idx, out=y_soft, args=args)
                result = [train_acc, val_acc, test_acc]
                logger.add_result(runs_overall, result)
                print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
            t_end = time.perf_counter()
            duration = t_end - t_start
            if args.log_steps > 0:
                print(print(f'Split: {split + 1:02d}, 'f'Run: {run + 1:02d}'), 'time: ', duration)
                logger.print_statistics(runs_overall)
    print('run time now:', time.time()-start)
    total_end = time.perf_counter()
    total_duration = total_end - total_start
    print('total time: ', total_duration)
    logger.print_statistics()
    train1_acc, valid_acc, train2_acc, test_acc, \
    train1_var, valid_var, train2_var, test_var = logger.best_result(run=None, with_var=True) # to adjust

    if trial is not None:
        trial.set_user_attr("train", train2_var)
        trial.set_user_attr("valid", valid_var)
        trial.set_user_attr("test", test_var)

    return valid_acc

def set_up_trial(trial: optuna.Trial, args):
    args.lr     = trial.suggest_float('lr', 1e-3, 3e-2, log=True)
    args.weight_decay     = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    args.dropout     = trial.suggest_float('dropout', 0, 0.95, step=0.05)
    args.loss = trial.suggest_categorical("loss", ["CE", "MSE"])
    if args.ogb:
        args.hidden_channels = trial.suggest_int("hidden_channels", 64, 256, step=64)
        args.num_layers = trial.suggest_int("num_layers", 1, 3)
    else:
        args.hidden_channels = trial.suggest_int("hidden_channels", 64, 64, step=16)
        args.num_layers = trial.suggest_int("num_layers", 1, 2)
    if args.model == 'LP':
        args.alpha = trial.suggest_uniform('alpha', 0, 1.00001)
    elif args.model in ['APPNP', 'IAPPNP', 'MLP']:
        args.alpha     = trial.suggest_uniform('alpha', 0, 1.00001)
        args.pro_alpha = trial.suggest_uniform('pro_alpha', 0, 1.00001)
        args.K = trial.suggest_uniform('K', 0, 1000)

    elif args.model in ['ElasticGNN', 'ALTOPT', 'ORTGNN', "EXACT", "AGD"]:
        args.alpha = trial.suggest_float('alpha', 0, 1.00001, step=0.05)
        args.loop = trial.suggest_int('loop', 1, 1)
        args.K = trial.suggest_int("K", 1, 3)
        args.K0 = trial.suggest_int("K0", 1, 30)
        args.lambda1 = trial.suggest_float('lambda1', 0, 20, step=0.05)
        args.lambda2 = trial.suggest_float('lambda2', 0, 20, step=0.05)
        args.useGCN = trial.suggest_categorical("useGCN", [False])
        args.softmaxF = trial.suggest_categorical("softmaxF", [True, False])
        args.Fwd     = trial.suggest_float('Fwd', 1e-6, 1e-1, log=True)
        args.gnnepoch = trial.suggest_int("gnnepoch", 0, 60, step=10)
        args.weightedloss = trial.suggest_categorical("weightedloss", [False])
        args.temperature = trial.suggest_float("temperature", 0.01, 10, log=True)
        args.bn = trial.suggest_categorical("bn", [True])
        args.tailln = trial.suggest_categorical("tailln", [True])
    
    elif args.model in ['MFGNN', 'MFGNN-Hidden']:
        args.lambda1 = trial.suggest_uniform('lambda1', 0, 1000)
        args.K = trial.suggest_uniform('K', 0, 1000)
        print('lambda1: ', args.lambda1)

    elif args.model in ['CS']:
        args.num_correct_layer = trial.suggest_uniform('num_correct_layer', 0, 100)
        args.num_smooth_layer = trial.suggest_uniform('num_smooth_layer', 0, 100)
        args.correct_alpha = trial.suggest_uniform('correct_alpha', 0, 1.0001)
        args.smooth_alpha = trial.suggest_uniform('smooth_alpha', 0, 1.0001)
        args.alpha = trial.suggest_uniform('alpha', 0, 1.00001)
    print('K: ', args.K)
    print('alpha: ', args.alpha)
    print('lr: ', args.lr)
    print('weight_decay: ', args.weight_decay)
    print('dropout: ', args.dropout)
    return args


if __name__ == "__main__":
    optuna_total_start = time.perf_counter()

    args = parse_args()
    if args.test:
        objective()
    else:
        study = optuna.create_study(storage=f"sqlite:///AGD/{args.dataset}_{args.model}_{args.fix_num}_{args.proportion}.db", study_name=f"{args.dataset}_{args.model}", direction="maximize",  load_if_exists=True)

        study.optimize(objective, n_trials=1500)

        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        sorted_trial = sort_trials(study.trials, key=args.sort_key)

        for trial in sorted_trial:
            print("trial.params: ", trial.params, 
                "  trial.value: ", '{0:.5g}'.format(trial.value),
                "  ", trial.user_attrs)

        test_acc = []
        for trial in sorted_trial:
            test_acc.append(trial.user_attrs['test'])
        print('test_acc')
        print(test_acc)

        print("Best params:", study.best_params)
        print("Best trial Value: ", study.best_trial.value)
        print("Best trial Acc: ", study.best_trial.user_attrs)

        optuna_total_end = time.perf_counter()
        optuna_total_duration = optuna_total_end - optuna_total_start
        print('optuna total time: ', optuna_total_duration)

