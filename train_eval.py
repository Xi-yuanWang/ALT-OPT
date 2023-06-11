import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
import  math
import torch.nn as nn
from torch_geometric.data import Data as PygData

def cross_entropy(pred, target):
    return torch.mean(torch.sum(-target * pred, 1))

def KL(pred, target):
    return F.kl_div(pred, target)


def train_altopt(model: nn.Module, data: PygData, train_idx: torch.Tensor, optimizer, args=None):
    y = data.y
    model.train()
    optimizer.zero_grad()
    out = model(data=data)
    label = model.FF

    if label is None:
        label = model.prop.init_label(data)
    train_mask = data.train_mask
    total_weight = torch.zeros(y.shape[0]).cuda()
    total_weight[train_mask] = 1

    if args.current_epoch == 0:
        num = 0
    else:
        num = 100
    if args.current_epoch > 0:
        if args.weightedloss:
            tlabel = F.softmax(label/args.temperature, dim=-1)
            weight = 1 - torch.sum(-tlabel * torch.log(tlabel.clamp(min=1e-8)), 1) / math.log(args.num_class)
            index = label.argmax(dim=1)
            for i in range(args.num_class):
                tweight = weight.clone()
                pos = (index == i)
                tweight[~pos] = 0
                tweight[train_mask] = 0
                value, indices = torch.topk(tweight, num)
                total_weight[indices] = value
        else:
            total_weight.fill_(1)
    if args.loss == "MSE":
        if args.softmaxF:
            diff = torch.square(out - F.softmax(label/args.temperature, dim=-1))
        else:
            diff = torch.square(out - label)
    elif args.loss == "CE":
        if args.softmaxF:
            diff = - F.softmax(label/args.temperature, dim=-1) * out # - label * out  
        else:
            diff = - label * out
    diff = torch.sum(diff, 1) 
    loss = torch.sum(total_weight * diff)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test_altopt(model, data, split_idx, args=None):
    model.eval()
    if args.model == 'ALTOPT':
        out = model(data=data) 
        out = model.FF
    else:
        out = model(data=data)
    
    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1) # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    return train_acc, valid_acc, test_acc

def train(model, data, train_idx, optimizer, args=None):
    # print('train')
    model.train()
    torch.autograd.set_detect_anomaly(True) ## to locate error of NaN
    optimizer.zero_grad()
    out = model(data=data)[train_idx]
    
    if len(data.y.shape) == 1:
        y = data.y[train_idx]
    else:
        y = data.y.squeeze(1)[train_idx]  ## for ogb data

    if args.loss == 'CE':
        loss = F.nll_loss(out, y)
    elif args.loss == 'MSE':
        # convert y to one-hot format
        label = torch.zeros_like(out) 
        label[range(y.shape[0]), y] = 1
        # import ipdb; ipdb.set_trace()
        loss = torch.sum(torch.square(out-label))
    # print('####### training loss: ', loss)
    loss.backward()
    # print('loss: ', loss)
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, args=None):
    # print('test')
    model.eval()
    # if args.model == 'ALTOPT':
    #     out = model.FF ## hidden
    # else:
    #     out = model(data=data)
    out = model(data=data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1) # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    # y_true_test = y[split_idx['test']]
    # y_pred = y_pred[split_idx['test']]
    # if args.current_epoch % 100 == 0:
    #     for i in range(7):
    #         pos = (y_pred==i)
    #         y_true_i = y_true_test[pos]
    #         y_pred_i = y_pred[pos]
    #         print(i, torch.sum(y_pred_i==y_true_i)/len(y_pred_i))
    
    return train_acc, valid_acc, test_acc
    # return out, train_acc, valid_acc, test_acc
    # return -train_loss, -valid_loss, -test_loss


def train_appnp(model, data, train_idx, optimizer, args=None):
    # print('train')
    model.train()
    torch.autograd.set_detect_anomaly(True)  ## to locate error of NaN
    optimizer.zero_grad()
    out, loss1 = model(data=data)
    out = out[train_idx]
    # out = model(data=data)[train_idx]

    if len(data.y.shape) == 1:
        y = data.y[train_idx]
    else:
        y = data.y.squeeze(1)[train_idx]  ## for ogb data

    if args.loss == 'CE':
        if args.current_epoch > 100:
            loss = F.nll_loss(out, y) + 0.005 * loss1
            # loss = F.nll_loss(out, y)
        else:
            loss = F.nll_loss(out, y)
        # loss = F.nll_loss(out, y)
    elif args.loss == 'MSE':
        # convert y to one-hot format
        label = torch.zeros_like(out)
        label[range(y.shape[0]), y] = 1
        # import ipdb; ipdb.set_trace()
        loss = torch.pow(torch.norm(out - label), 2)
    # print('####### training loss: ', loss)
    loss.backward()
    # print('loss: ', loss)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_appnp(model, data, split_idx, args=None):
    # print('test')
    model.eval()
    # if args.model == 'ALTOPT':
    #     out = model.FF ## hidden
    # else:
    #     out = model(data=data)
    out, diff = model(data=data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1)  # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    # y_true_test = y[split_idx['test']]
    # y_pred = y_pred[split_idx['test']]
    # if args.current_epoch % 100 == 0:
    #     for i in range(7):
    #         pos = (y_pred==i)
    #         y_true_i = y_true_test[pos]
    #         y_pred_i = y_pred[pos]
    #         print(i, torch.sum(y_pred_i==y_true_i)/len(y_pred_i))

    return train_acc, valid_acc, test_acc


@torch.no_grad()
def test1(model, data, out, split_idx, args=None):
    # print('test')
    model.eval()
    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1)  # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    print('base model test_acc', test_acc)
    return train_acc, valid_acc, test_acc


def train_cs(model, data, train_idx, optimizer, args=None):
    model.train()
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()
    out = model(data=data)[train_idx]
    out = F.log_softmax(out, dim=-1)
    if len(data.y.shape) == 1:
        y = data.y[train_idx]
    else:
        y = data.y.squeeze(1)[train_idx]  ## for ogb data
    loss = F.nll_loss(out, y)
    # loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test_cs(model, data, split_idx, out=None, args=None):
    model.eval()
    if out is None:
        out = model(data=data)
    y_pred = out.argmax(dim=-1, keepdim=True)
    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1) # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    return train_acc, valid_acc, test_acc, out
