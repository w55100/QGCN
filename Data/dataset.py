import sys
import os
import pickle
import numpy as np
from scipy import sparse
import torch
import networkx as nx

from Model.matrix_utils import get_hatA,coo2torchsparse

"""
在原实现中。
先对feature矩阵进行归一化。 feature(NxD)每行除以行加总???
然后是Adj处理得到hatA。


这个训练方式一开始让人觉得迷茫。
差别就在于选择inductive还是transductive。

inductive是训练时看不到测试集数据。
transductive是训练时带着测试集数据训练，但是只给出训练集的标签。

在本例中，可以看到喂入模型的features是全量的（含测试集的结点的属性值）。
也就是说模型在训练时已经见过这些测试集数据了。
显然训练方式是transductive。

那多问一句，可以不可以弄成inductive呢？
猜想在图的半监督训练中是没有必要的。
因为强行隐藏测试集结点，意味着破坏图结构。起码邻接矩阵里，对应测试集结点的连边都得消掉？

这就可以理解了，为什么给模型全量feature。
但是根据不同阶段，只取不同的index算maskloss。
"""


def sample_mask(idx, l):
    """
    Create mask.
    source:https://github.com/dragen1860/GCN-PyTorch/blob/master/data.py
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def row_normalize(mx):
    """Row-normalize sparse matrix
    from https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
    this is a uncomfortable function,
    since divided by zero is not welcomed.
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten() #RuntimeWarning: divide by zero encountered in power
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def row_normalize_safe(features):
    """ safe version.
    dived by the sum of each row.
    input : [N,D] np.ndarray
    """
    rsum = features.sum(axis=1) #(N,)
    rsum[rsum!=0] = 1/ rsum[rsum!=0]
    # some row may have pos and neg numbers,
    # but the rowsum happened to be zero.
    # in this case we dont change the numbers.
    rsum[rsum==0] = 1
    ans = features.T*rsum #(D,N)*(N,) fit broadcast rule
    return ans.T

def prepare_data(dataset_dir,dataset_name):
    # data load
    # #py3中想读取py2中保存的数据，需要指定编码
    # x=pickle.load(open('Data/citeseer/ind.citeseer.x','rb'),encoding='iso-8859-1')#(120, 3703)
    # y=pickle.load(open('Data/citeseer/ind.citeseer.y','rb'),encoding='iso-8859-1')#(120, 6),onehot

    print("Loading raw data from files...")

    #dataset_dir = './Data'
    #dataset_name = 'citeseer'
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(dataset_dir+"/ind.{}.{}".format(dataset_name, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pickle.load(f, encoding='latin1'))
            else:
                objects.append(pickle.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    print("Done.")

    print("Processing Data...")

    index = []
    for line in open(dataset_dir+"/ind.{0}.test.index".format(dataset_name)):
        index.append(int(line.strip()))

    test_idx_reorder = index
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_name == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)  # [2312,3326]
        tx_extended = sparse.lil_matrix((len(test_idx_range_full), x.shape[1]))  # shape=(1015,3703)
        tx_extended[test_idx_range - min(test_idx_range), :] = tx  # 没有数据的序号当成属性全0。
        # 注意上一行用的是test_idx_range，所以现在tx内部还不是按照idx升序排列的。
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))  # shape=(1015,6)
        ty_extended[test_idx_range - min(test_idx_range), :] = ty  # 没有数据的序号当成标签全0。
        ty = ty_extended

        # 注意这步结束之后，tx与ty中，与idx相吻合的行号，都是对应的值。

    features = sparse.vstack([allx, tx]).toarray() #(3327,3703)
    features[test_idx_reorder, :] = features[test_idx_range, :]  # 这一步保证了feature内，test结点每一行的行号都与其值对应的index吻合。

    ### 对feature归一化。
    features = row_normalize_safe(features)

    labels = np.vstack((ally, ty)) #(N,n_classes) , (3327,6)
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # 同理,行号与index对齐。

    idx_test = test_idx_range.tolist() #从这步可以看到空白孤立点并不参与验证集的检验，index不在名单中。
    idx_train = range(len(y))  # range(120)
    idx_val = range(len(y), len(y) + 500)  # range(120,620)

    # 在citeseer数据集里有[0,119]个点在x中，[0,2311]共2312个点在allx中，有[2312,2326]共1015个点在tx中。
    # 不知道为什么训练集120，验证集只划分了500个。

    ###图结构
    # 这里偷懒用了nx包的函数，坏处是多引进了一个包却只为了一个函数。
    # 可以考虑自己写一个从dict创建adjacent matrix的函数。
    adj_matrix = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    hatA = torch.FloatTensor(get_hatA(adj_matrix))#(3327,3327)

    print('Data processed to numpy.')
    # 目前为止都是numpyd ，处理为适配torch模型的形态

    n_classes = labels.shape[1] #(N,cls)
    labels = torch.LongTensor(labels).argmax(dim=1) #(N,)
    features = torch.FloatTensor(features) #(N,D)

    print('Data adjusted to torch.')

    return hatA,features,labels, n_classes,idx_train,idx_val,idx_test