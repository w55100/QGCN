import sys
import pickle
import numpy as np
import torch
import argparse
import yaml
import networkx as nx

from qtrainer import GCNTrainer
from Model.GCN import GCN
from Data.dataset import prepare_data

"""


"""


def freeze_rand(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=996007, help='')
    parser.add_argument('--dataset', type=str, default='citeseer', help='')

    parser.add_argument('--n_layers', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=1024,  help='')

    parser.add_argument('--n_epochs', type=int, default=500, help='')
    parser.add_argument('--lr', type=float, default=0.001,  help='')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='param in Adam and SGD optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='param in SDG optimizer')
    parser.add_argument('--optimizer', type=lambda x: str(x).lower(), default='adam', help='')

    parser.add_argument('--save_interval', type=int, default=10, help='')

    args = parser.parse_args()
    return args


def masked_loss(out, label, mask):
    loss = F.cross_entropy(out, label, reduction='none')
    loss *= mask
    loss = loss.sum() / mask.sum()  # 1/n
    return loss


def masked_acc(out, label, mask):
    # [node, f]
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    correct *= mask
    acc = correct.sum() / mask.sum()
    return acc


def cal_acc(pred, label):
    """input: torch.Tensor"""
    pred = pred.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    acc = correct.mean()
    return acc


def switch_optimizer(optimizer,model,args):
    if optimizer == 'adam':
        Optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer == 'sgd':
        Optim = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    else:
        raise ValueError('unknown optimizer type {}'.format(optimizer))

    return Optim

def train(args, data):
    # hatA, features, train_label, val_label, test_label, train_mask, val_mask, test_mask = data
    hatA, features, labels, n_classes , idx_train, idx_val, idx_test = data

    # cuda shift
    device = torch.device('cuda')
    hatA = hatA.to(device)
    features = features.to(device)
    labels = labels.to(device)

    model = GCN(
        hatA,
        n_classes= n_classes,
        n_layers=args.n_layers,
        inpt_dim=features.shape[1],  # 3703,
        hidden_dim=args.hidden_dim,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device)

    optimizer = switch_optimizer(args.optimizer,model,args)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

    dataloader = [(features, labels, idx_train)]  # 保证是可迭代类型，每次迭代一份data

    trainer = GCNTrainer(
        dataloader, model, criterion, optimizer,
        validation=masked_acc,
        n_epochs=args.n_epochs,
        save_dir='./Result',
        save_interval=args.save_interval,
        device=device,
        metric=cal_acc,
        idx_val = idx_test,
        label_val = labels[idx_test]
        #idx_val=idx_val,
        #label_val=labels[idx_val]
    )
    trainer.train()
    trainer(manner='classic')

    # #after train, go to test
    # pred = model(features)
    # test_acc = cal_acc(pred[idx_test], labels[idx_test])
    # print('test in {0} acc= {1:.4f}'.format(args.dataset,test_acc))


if __name__ == '__main__':
    args = get_args()
    freeze_rand(args.seed)
    data = prepare_data(dataset_dir='./Data',dataset_name=args.dataset)
    train(args, data)
