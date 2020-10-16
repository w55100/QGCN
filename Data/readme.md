
## .content

3312 papers
3703 words
$X \in R^{3312 \times 3703}$


---
6 classes{
			Agents
			AI
			DB
			IR
			ML
			HCI
}
---
each line
`<paper_id> <word_attributes>+ <class_label>`




## .cite
`<ID of cited paper> <ID of citing paper>`


## 8个分割后的文件
ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances 
    (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    
ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;

ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict object;
ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

All objects above must be saved using python pickle module.


You can use cPickle.load(open(filename)) to load the numpy/scipy objects x, y, tx, ty, allx, ally, 
and graph. test.index is stored as a text file.