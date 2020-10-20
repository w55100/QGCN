
This is a reproduction of GCN(Graph Convolution Network) via PyTorch(>1.0)

>paper:
>Thomas N. Kipf,.etc, [Semi-Supervised Classification with Graph Convolutional Networks, ICLR2017](https://arxiv.org/pdf/1609.02907.pdf)
>
>official repo: https://github.com/tkipf/gcn/ (tensorflow0.12)
>
>official repo: https://github.com/tkipf/pygcn (torch0.4.0)


Same as original paper, we keep the data-preprocessing manner of 
Zhilin Yang,.etc, [Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/pdf/1603.08861.pdf).  [(repo)](https://github.com/kimiyoung/planetoid)
>so we get 8 files named as 'ind.{dataset_name}.{data_name}'
>
>for data_name in [x,y,allx,ally,tx,ty,graph,test.index]   

# Requirements：
- torch 1.6.0
- networkx 2.5

# Benchmark


| dataset       | Citeseea | Cora | Pubmed | NELL |
|---------------|----------|------|--------|------|
| GCN(paper)    | 70.3     | 81.5 | 79.0   | 66.0 |
| This repo     | 72.3     | 81.3 | 80.3   |      |



# Best Run

citeseer 72.3
```bash
python train.py --dataset citeseer --n_epochs 500 --lr 0.001 --optimizer adam
```


cora 81.3
```bash
 python train.py --dataset pubmed --n_epochs 1000 --save_interval 100 --lr 0.001 --optimizer adam
```

pubmed 80.3
```bash
python train.py --dataset pubmed --n_epochs 6000 --save_interval 100 --lr 0.0001 --optimizer adam
```