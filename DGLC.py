# Optional: eliminating warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from arguments import arg_parse
from evaluate_embedding import evaluate_embedding
from gin import Encoder
from losses import local_global_loss_
from model import FF, PriorDiscriminator, Cluster
from torch import optim
from torch.autograd import Variable
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import json
import numpy as np
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from evaluate_embedding import cluster_acc
from sklearn import preprocessing
import math
import csv

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

class InfoGraph(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(InfoGraph, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)

    self.cluster_embedding = Cluster(args.hidden_dim * args.num_gc_layers, args.cluster_emb)
    self.cluster_layer = Parameter(torch.Tensor(args.cluster_emb, args.cluster_emb))   
    torch.nn.init.xavier_uniform_(self.cluster_layer.data)

    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

    self.alpha = 1.0

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

  def get_results(self, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding = []
    cluster = []
    y = []
    with torch.no_grad():
        for data in loader:
            data.to(device)
            x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
            if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
            z, q, _, _ = self.forward(x, edge_index, batch, num_graphs)
            embedding.append(z.cpu().numpy())
            cluster.append(q.cpu().numpy())
            y.append(data.y.cpu().numpy())
    embedding = np.concatenate(embedding, 0)
    cluster = np.concatenate(cluster, 0)
    y = np.concatenate(y, 0)
    return embedding, cluster, y

  def get_p(self, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cluster = []
    with torch.no_grad():
        for data in loader:
            data.to(device)
            x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
            if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
            z, q, _, _ = self.forward(x, edge_index, batch, num_graphs)
            cluster.append(q.cpu().numpy())
    cluster = torch.from_numpy(np.concatenate(cluster, 0))
    p_distribution = target_distribution(cluster).to(device)
    return p_distribution


  def forward(self, x, edge_index, batch, num_graphs):
    if dataset.data.x is None or np.shape(dataset.data.x)[1] == 0:
        x = torch.ones(batch.shape[0],1).to(device)

    y, M = self.encoder(x, edge_index, batch)
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)
    
    # Clustering layer
    z = self.cluster_embedding(y)
    q = 1.0 / (1.0 + torch.sum(
        torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
    q = q.pow((self.alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()

    return z, q,  g_enc, l_enc


if __name__ == '__main__':
    import time
    start = time.time() 
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = arg_parse()
    accuracies = {'acc':[], 'nmi':[], 'ari':[], 'randomforest':[]}
    epochs = 20
    log_interval = 1
    batch_size = 128
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    dataset = TUDataset(path, name=DS).shuffle()
    dataset_num_features = max(dataset.num_features, 1)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    print('================')
    print('Dataset:', DS)
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('clutering embedding dimension: {}'.format(args.cluster_emb))
    print('================')

    iter = 1

    for it in range(iter):
        model = InfoGraph(args.hidden_dim, args.num_gc_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        mode='fd'
        measure='JSD'
        #'GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1
        accmax = 0
        nmimax = 0
        arimax = 0

        for epoch in range(1, epochs+1):
            loss_all = 0
            batch = 0

            if epoch == 3:
                model.eval()
                emb, _, y = model.get_results(dataloader)
                n_cluster = len(np.unique(y))
                print("class: ", n_cluster)
                print('===== Start training =====')
                kmeans = KMeans(n_clusters=n_cluster, n_init=100)
                y_pred = kmeans.fit_predict(emb)
                model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

            model.train()

            for data in dataloader:
                data = data.to(device)
                optimizer.zero_grad()
                _, q, g_enc, l_enc = model(data.x, data.edge_index, data.batch, data.num_graphs)

                local_global_loss = local_global_loss_(l_enc, g_enc, data.edge_index, data.batch, measure)


                if epoch >= 3:
                    p = target_distribution(q)
                    kl_loss = F.kl_div(q.log(), p)
                    loss = local_global_loss + kl_loss 
                    batch += 1  
                else:
                    loss = local_global_loss
                # Total loss

                loss_all += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()

            print('===== Epoch {}, Loss {} ====='.format(epoch, loss_all / len(dataloader)))
            
            if epoch % log_interval == 0:
                model.eval()
                emb, q, y = model.get_results(dataloader)
                y_pred = q.argmax(1)
                acc = cluster_acc(y, y_pred)
                nmi = nmi_score(y, y_pred)
                ari = ari_score(y, y_pred)
                if acc > accmax:
                    accmax = acc
                if nmi > nmimax:
                    nmimax = nmi
                if ari > arimax:
                    arimax = ari

                print('===== Clustering performance: =====')
                print('Iter {}'.format(epoch), ':Acc {:.4f}'.format(acc),
                    ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari), 
                    ':Accmax {:.4f}'.format(accmax), ', nmimax {:.4f}'.format(nmimax), 
                    ', arimax {:.4f}'.format(arimax))
