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
    parser.add_argument('--log_steps', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--normalize_features', type=str2bool, default=True, help="whether to normalize node feature")
    parser.add_argument('--random_splits', type=int, default=0, help='default use fix split')

    parser.add_argument('--model', type=str, default='ALTOPT')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)

    parser.add_argument('--prop', type=str, default='EMP') # useless
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
    return args

def objective(trial):
    args = parse_args()
    args = set_up_trial(trial, args)
    print(args)

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
            model.propagate(data)
            print('propagate done')

        for run in range(args.runs):
            data.pseudo_mask = data.train_mask.clone()
            data.pseudo_label = data.y.clone()

            data.total_weight = torch.zeros(data.y.shape[0]).cuda()
            data.f = None
            runs_overall = split * args.runs + run
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            t_start = time.perf_counter()
            best_acc = 0
            y_soft = None

            args.current_epoch = 0
            if args.model in ['ALTOPT', "EXACT", "AGD"]:
                for i in range(100):
                    loss = train_altopt(model, data, train_idx, optimizer, args=args)

                result = test(model, data, split_idx, args=args)
                print('test_result', result)
                args.current_epoch = 1
            for epoch in range(1, 1 + args.epochs):
                args.current_epoch = epoch
                if args.model in ['ALTOPT', "EXACT", "AGD"]:
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

    trial.set_user_attr("train", train2_var)
    trial.set_user_attr("valid", valid_var)
    trial.set_user_attr("test", test_var)

    return valid_acc

def set_up_trial(trial: optuna.Trial, args):
    args.lr     = trial.suggest_uniform('lr', 0, 1)
    args.weight_decay     = trial.suggest_uniform('weight_decay', 0, 1)
    args.dropout     = trial.suggest_uniform('dropout', 0, 1)
    if args.loss is None:
        args.loss = trial.suggest_categorical("loss", ["CE", "MSE"])

    if args.model == 'GCN':
        pass
    elif args.model == 'GAT':
        pass
    elif args.model == 'LP':
        args.alpha = trial.suggest_uniform('alpha', 0, 1.00001)
    elif args.model in ['APPNP', 'IAPPNP', 'MLP']:
        args.alpha     = trial.suggest_uniform('alpha', 0, 1.00001)
        args.pro_alpha = trial.suggest_uniform('pro_alpha', 0, 1.00001)
        args.K = trial.suggest_uniform('K', 0, 1000)

    elif args.model in ['ElasticGNN', 'ALTOPT', 'ORTGNN', "EXACT", "AGD"]:
        if True:
            args.lambda1 = trial.suggest_uniform('lambda1', 0, 1)
            args.lambda2 = trial.suggest_uniform('lambda2', 0, 10)
            args.alpha = 0 #trial.suggest_uniform('alpha', 0, 1.00001)
            print('lambda1: ', args.lambda1)
            print('lambda2: ', args.lambda2)
            args.loop = trial.suggest_uniform('loop', 0, 10)

        elif args.prop == 'CP':
            args.alpha = trial.suggest_uniform('alpha', 0, 1.00001)

        args.K = trial.suggest_uniform('K', 0, 1000)
    
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

def set_up_search_space(args):
    dropout_range = [args.dropout]
    lr_range = [args.lr]
    wd_range = [args.weight_decay]
    alpha_range = [args.alpha]
    lambda1_range = [args.lambda1]
    lambda2_range = [args.lambda2]
    K_range = [args.K]
    loop = [args.loop]
    num_correct_layer_range = [args.num_correct_layer]
    correct_alpha_range = [args.correct_alpha]
    num_smooth_layer_range = [args.num_smooth_layer]
    smooth_alpha_range = [args.smooth_alpha]
    pro_alpha_range = [args.pro_alpha]
    if args.loss is None:
        loss_range = ["CE", "MSE"]
    else:
        loss_range = [args.loss]
    if args.loop is None:
        loop = [1]
    if args.dropout is None:
        dropout_range = [0.5, 0.8]

    if args.lr is None:
        lr_range = [0.1, 0.01, 0.05, 0.001, 0.0005, 0.0001]

    if args.weight_decay is None:
        wd_range = [5e-3, 5e-4, 5e-5]
        wd_range = [5e-4, 5e-5]

    if args.model == 'LP':
        if args.alpha is None:
            alpha_range = [0.7, 0.8, 0.9, 1, 1.1]

    if args.model == 'APPNP' or args.prop == 'APPNP' or args.model == 'IAPPNP' or args.model == 'MLP':
        if args.alpha is None:
            alpha_range = [0, 0.05, 0.1, 0.15] 
            alpha_range = [0, 0.1, 0.2]
            if 'adv' in args.dataset:
                alpha_range = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
        if args.pro_alpha is None:
            pro_alpha_range = [0, 0.1, 0.3, 0.8, 1, 1.5, 2]
        if args.K is None:
            K_range = [5, 10]

    if args.model in ['ElasticGNN'] and args.prop == 'EMP':
        range_list = [0, 3, 6, 9, 15]
        if args.lambda1 is None:
            lambda1_range = range_list
        
        if args.lambda2 is None:
            lambda2_range = range_list

        if args.lambda1 is None and args.lambda2 is None:
            range_list = [0, 3, 6, 9] ## L1+L2
            lambda1_range = range_list
            lambda2_range = range_list

        if args.K is None:
            K_range = [5, 10]
            # K_range = [1, 3, 5, 7, 10]
            # K_range = [0, 2, 4, 6, 8, 10, 12]
    
    if args.model in ['MFGNN', 'MFGNN-Hidden']:
        # range_list = [0, 3, 6, 9, 15]
        range_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        if args.lambda1 is None:
            lambda1_range = range_list

        if args.K is None:
            K_range = [5, 10]


    if args.model in ['ElasticGNN'] and args.prop == 'CP':
        if args.alpha is None:
            alpha_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    if args.model in ['ALTOPT', "AGD"]:
        range_list = [0.01, 0.02, 0.05, 0.1, 0.2]

        if args.lambda1 is None:
            lambda1_range = range_list
        
        if args.lambda2 is None:
            lambda2_range = [0, 0.01, 0.05, 0.1, 0.2, 0.3]

        if args.lambda1 is None and args.lambda2 is None:
            range_list = [0.01, 0.1, 0.4, 0.5, 0.6, 0.8]

            lambda1_range = [0.1, 0.3, 0.5, 1]
            lambda2_range = [1, 3, 5, 10]
        if args.alpha is None:
            alpha_range = [0.3, 0.5, 0.7, 0.9]
        if args.K is None:
            K_range = [1, 5, 10, 20, 50]
        if args.loop is None:
            loop = [1, 5, 10]

    if args.model in ['EXACT']:
        range_list = [0.01, 0.02, 0.05, 0.1, 0.2]

        if args.lambda1 is None:
            lambda1_range = range_list
        
        if args.lambda2 is None:
            lambda2_range = [0, 0.01, 0.05, 0.1, 0.2, 0.3]

        if args.lambda1 is None and args.lambda2 is None:
            range_list = [0.01, 0.1, 0.4, 0.5, 0.6, 0.8]
            lambda1_range = [0.1, 0.3, 0.5, 1]
            lambda2_range = [1, 3, 5, 10]
        if args.alpha is None:
            alpha_range = [0]
        K_range = [10]
        if args.loop is None:
            loop = [1, 5, 10]

    if args.model in ['ORTGNN']:
        alpha_range = [0]
        range_list = [0.01, 0.02, 0.05, 0.1, 0.2]

        if args.lambda1 is None:
            lambda1_range = range_list

        if args.lambda2 is None:
            lambda2_range = [1, 5, 10]

        if args.lambda1 is None and args.lambda2 is None:
            range_list = [0.01, 0.1, 0.5]

            lambda1_range = [0.5, 1, 5]
            lambda2_range = range_list

        if args.K is None:
            K_range = [1, 5, 10]

    if args.model in ['CS']:
        if args.num_correct_layer is None:
            num_correct_layer_range = [20, 50]
        if args.correct_alpha is None:
            correct_alpha_range = [1, 0.9, 0.5, 0.3, 0.1, 0]
        if args.num_smooth_layer is None:
            num_smooth_layer_range = [20, 50]
        if args.smooth_alpha is None:
            smooth_alpha_range = [0.8, 0.6, 0.3, 0.1, 0]
        if args.alpha is None:
            alpha_range = [0, 0.1, 0.3, 0.5, 0.8, 0.9, 1]

    search_space = {"lr": lr_range, 
                    "weight_decay": wd_range, 
                    "lambda1": lambda1_range,
                    "lambda2": lambda2_range, 
                    "alpha": alpha_range, 
                    "dropout": dropout_range,
                    "K": K_range,
                    'loop': loop,
                    "num_correct_layer": num_correct_layer_range,
                    "correct_alpha": correct_alpha_range,
                    "num_smooth_layer": num_smooth_layer_range,
                    "smooth_alpha": smooth_alpha_range,
                    "pro_alpha": pro_alpha_range,
                    "loss": loss_range
                    }
    return search_space

if __name__ == "__main__":
    optuna_total_start = time.perf_counter()

    args = parse_args()
    print('main: ', args)
    search_space = set_up_search_space(args)
    print('search_space: ', search_space)
    num_trial = 1
    for s in search_space.values():
        num_trial = len(s) * num_trial
    print('num_trial: ', num_trial)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GridSampler(search_space))

    study.optimize(objective, n_trials=num_trial)

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

