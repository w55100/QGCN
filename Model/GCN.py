import torch
import torch.nn as nn
import pickle
import numpy as np

"""
记录trick
1.feature要行normalize
2.每层GCN有一个可选的bias
3.手动初始化参数

"""


class GCNLayer(torch.nn.Module):
    """
    input:[N,D] or [B,N,D]
    output:[N,F] or [B,N,F]
    """

    def __init__(self, indim, outdim, bias=True):
        super(GCNLayer, self).__init__()

        self.indim = indim
        self.outdim = outdim
        # 这样定义的theta可能全0就不存在训练可能了。
        self.theta = torch.nn.Parameter(torch.FloatTensor(indim, outdim))
        stdv = 1. / np.sqrt(self.theta.shape[1])
        self.theta.data.uniform_(-stdv, stdv)

        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(outdim))
            self.bias.data.uniform_(-stdv,stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, hatA, X):
        """ math matrix mul=mm"""
        # print(hatA.shape, X.shape, self.theta.shape)
        # print(torch.typename(hatA))
        # print(torch.typename(X))

        o = torch.mm(hatA, X)
        o = o.mm(self.theta)

        return o if self.bias is None else o + self.bias

    def train(self, mode=True):
        """覆盖默认方法"""
        self.training = mode
        if mode:
            self.theta.requires_grad = True
        else:
            self.theta.requires_grad = False

        return self

    def __repr__(self):
        return self.__class__.__name__ \
               + str(self.indim)  \
               + ' -> '+ str(self.outdim)


class GCN(torch.nn.Module):
    """
    原论文: https://arxiv.org/pdf/1609.02907.pdf
    GCN原作者用来解决图结点级别的多分类问题。
    """

    def __init__(self,
                 hatA,
                 n_classes,
                 n_layers=3,
                 inpt_dim=3703,
                 hidden_dim=1024):
        super(GCN, self).__init__()
        self.hatA = hatA
        self.n_classes = n_classes
        assert n_layers > 1
        self.n_layers = n_layers
        self.inpt_dim = inpt_dim
        self.hidden_dim = hidden_dim

        self.gclayers = nn.ModuleList([GCNLayer(inpt_dim, hidden_dim)] +
                                      [GCNLayer(hidden_dim, hidden_dim)
                                       for i in range(self.n_layers - 2)]
                                      + [GCNLayer(hidden_dim, n_classes)]
                                      )
        # self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x):
        # pickle.dump(self.hatA,open('./hatA.pkl','wb'))
        # pickle.dump(x, open('./x.pkl', 'wb'))
        # print('qq',torch.typename(self.hatA))
        for layer in self.gclayers:
            # print('hh', torch.typename(x))
            x = layer(self.hatA, x)

        # 最后一步是CE不需要自己做softmax
        # x = torch.nn.functional.softmax(x, dim=-1)
        return x
