import argparse

def arg_parse():
        parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
        parser.add_argument('--DS', dest='DS', help='Dataset')
        parser.add_argument('--local', dest='local', action='store_const', 
                const=True, default=False)
        parser.add_argument('--glob', dest='glob', action='store_const', 
                const=True, default=False)
        parser.add_argument('--prior', dest='prior', action='store_const', 
                const=True, default=False)
        parser.add_argument('--lr', dest='lr', type=float,
                help='Learning rate.')
        parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
                help='Number of graph convolution layers before each pooling')
        parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
                help='')
        parser.add_argument('--cluster_emb', dest='cluster_emb', type=int, default=10, help='')
        parser.add_argument('--d', dest='d', type=int, default=10, help='')
        parser.add_argument('--eta', dest='eta', type=int, default=2, help='')
        parser.add_argument('--clusters', dest='clusters', type=int, default=2, help='')    
        parser.add_argument('--preprocess', dest='preprocess', default=False) 
        parser.add_argument('--loss', dest='loss', default='kl') 
        return parser.parse_args()

