# Deep Graph-Level Clustering (DGLC)

This is an implement of the DGLC method in "Jinyu Cai, Yi Han, Wenzhong Guo, and Jicong Fan, [Deep graph-level clustering using pseudo-label-guided mutual information maximization network](https://link.springer.com/article/10.1007/s00521-024-09575-4), Neural Computing and Applications, 2024".

## Running Example:

  - You can run the model with some hyperparameters. For example:
    ```
    python DGLC.py --DS BZR --lr 0.00001 --num-gc-layers 4 --hidden-dim 32 --cluster_emb 25
    ```
  - The supported database is [TUDataset](https://chrsmrrs.github.io/datasets/docs/home/), which will be downloaded automatically at runtime (if it does not exist locally).

## Parameter description:
- DS is the dataset name. You can change from (MUTAG, PTC-MR, BZR, PTC-MM, ENZYMES, COX2), and the dataset will be downloaded automatically.
- num-gc-layers is the number of GNN hidden layers.
- hidden-dim is the dimension of the hidden layer of GNN.
- cluster_emb is the dimension of the cluster projector.

## Reference
If you use the code or find this repository useful for your research, please consider citing our paper.

```
@article{cai2024dglc,
  title = {Deep graph-level clustering using pseudo-label-guided mutual information maximization network},
  author = {Cai, Jinyu and Han, Yi and Guo, Wenzhong and Fan, Jicong},
  journal = {Neural Computing and Applications},
  pages = {1--16},
  year = {2024},
  publisher = {Springer}}
```


