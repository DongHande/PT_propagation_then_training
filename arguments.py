import argparse

# Training settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora_ml',
                        help='Choose from {cora_ml, citeseer, pubmed, ms_academic}')
    parser.add_argument('--K', type=int, default=10,
                        help='the depth of appnp and ptt when training')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='the alpha of appnp and ptt when training')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.005,
                        help='Set weight decay.')
    parser.add_argument('--loss_decay', type=float, default=0.05,
                        help='Set loss_decay.')
    parser.add_argument('--fast_mode', type=bool, default=False,
                        help='whether propogate when validation.')
    parser.add_argument('--mode', type=int, default=2,
                        help='For PT: 0-static(PTS); 1-dynamic(PTD); 2-adaptive(PTA).')
    parser.add_argument('--epsilon', type=int, default=100,
                        help='Set importance change of f(x).')
    parser.add_argument('--str_noise_rate', type=float, default=2.0, 
                        help='change the structure noise rate. Set it as 2.0 to keep the original noise rate.')
    parser.add_argument('--lbl_noise_num', type=int, default=0, 
                        help='change the lbl noise num. Set it as 0 to keep the original noise rate.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--seed', type=int, default=2144199737, help='Random seed for split data.')
    parser.add_argument('--ini_seed', type=int, default=2144199730, help='Random seed to initialize parameters.')
    return parser.parse_args()
