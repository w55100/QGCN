import yaml
import numpy as np
from numpy import linalg as la
from scipy import sparse
import torch

def coo2torchsparse(A):
    """Convert scipy.sparse.coo_matrix to torch.sparse.FloatTensor"""
    if not sparse.issparse(A):
        raise TypeError('input matrix should be scipy.sparse')
    if not sparse.isspmatrix_coo(A):
        A = A.tocoo()

    v = torch.FloatTensor(A.data)
    i = torch.LongTensor([A.row, A.col])
    shape = torch.Size(A.shape)

    res = torch.sparse.FloatTensor(i,v,shape)
    return res




def get_neg_haf(A):
    """
    求负二分之一次，先进行对角化A=QVQ^(-1)
    A^(-1/2)= QV^(-1/2)Q^(-1)
    """
    V,Q = np.linalg.eig(A)
    V = np.diag(V**(-0.5))
    R = np.dot(Q,V)
    R = np.dot(R,np.linalg.inv(Q))
    return R


def get_hatA(A):
    """
    $\hat{A}=\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$

    """
    h,w = A.shape
    assert h==w,"输入应该为方阵"

    tildeA = A + np .identity(h)  # 加边邻接阵,A为array则返回array，A为scipy则返回matrix
    if isinstance(tildeA,np.matrix):
        tildeA = tildeA.getA() #转为array
    tildeD = np.diag(np.sum(tildeA, axis=1).reshape(-1))  # 度数矩阵
    nhD = get_neg_haf(tildeD)
    hatA = np.dot(np.dot(nhD, tildeA), nhD)

    return hatA


def parse_yml(yml_path):
    if not os.path.isfile(yamlPath):
        raise FileNotFoundError("Cannot find %s" % yamlPath)
    cfg = yaml.load(open(yml_path,'r'))
    return cfg