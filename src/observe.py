import torch
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from earlystopping import EarlyStopping
from sample import Sampler
from metric import accuracy, roc_auc_compute_fn
# from deepgcn.utils import load_data, accuracy
# from deepgcn.models import GCN

from metric import accuracy
from utils import load_citation, load_reddit_data

from earlystopping import EarlyStopping
from sample import Sampler
import utils
import shutil
import os.path as osp

# Training settings
parser = argparse.ArgumentParser()
# Training parameter
parser.add_argument('--save_embeddings', type=int, default=0, help='save embeddings')
parser.add_argument('--alpha', type=float, default=1, help='alpha')
parser.add_argument('--write_res', type=int, default=0, help='if use step lr')
parser.add_argument('--selftraining', type=int, default=0, help='if use step lr')
parser.add_argument('--steplr', type=int, default=0, help='if use step lr')
parser.add_argument('--finetune', type=int, default=0, help='if plot')
parser.add_argument('--param_searching', type=int, default=0, help='if plot')
parser.add_argument('--train_size', type=int, default=0, help='if plot')
parser.add_argument('--pca', type=int, default=0, help='if plot')
parser.add_argument('--write_json', type=int, default=0, help='if plot')
parser.add_argument('--identity', type=int, default=0, help='if plot')
parser.add_argument('--plot_loss', type=int, default=0, help='if plot')
parser.add_argument('--load_pretrain', type=int, default=0, help='if unsupervised_mode')
parser.add_argument('--label_rate', type=float, default=0.05, help='if unsupervised_mode')
parser.add_argument('--unsupervised_mode', type=int, default=0, help='if unsupervised_mode')

parser.add_argument('--ssl', type=str, default=None, help='ssl agent')
parser.add_argument('--lambda_', type=float, default=0, help='if lploss')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Disable validation during training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=800,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.02,
                    help='Initial learning rate.')
parser.add_argument('--lradjust', action='store_true',
                    default=False, help='Enable leraning rate adjust.(ReduceLROnPlateau or Linear Reduce)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument("--mixmode", action="store_true",
                    default=False, help="Enable CPU GPU mixing mode.")
parser.add_argument("--warm_start", default="",
                    help="The model name to be loaded for warm start.")
parser.add_argument('--debug', action='store_true',
                    default=False, help="Enable the detialed training output.")
parser.add_argument('--dataset', default="cora", help="The data set")
parser.add_argument('--datapath', default="data/", help="The data path.")
parser.add_argument("--early_stopping", type=int,
                    default=0, help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")
parser.add_argument("--no_tensorboard", default=False, help="Disable writing logs to tensorboard")

# Model parameter
parser.add_argument('--type',
                    help="Choose the model to be trained.(mutigcn, resgcn, densegcn, inceptiongcn)")
parser.add_argument('--inputlayer', default='gcn',
                    help="The input layer of the model.")
parser.add_argument('--outputlayer', default='gcn',
                    help="The output layer of the model.")
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--withbn', action='store_true', default=False,
                    help='Enable Bath Norm GCN')
parser.add_argument('--withloop', action="store_true", default=False,
                    help="Enable loop layer GCN")
parser.add_argument('--nhiddenlayer', type=int, default=1,
                    help='The number of hidden layers.')
parser.add_argument("--normalization", default="AugNormAdj",
                    help="The normalization on the adj matrix.")
parser.add_argument("--sampling_percent", type=float, default=1.0,
                    help="The percent of the preserve edges. If it equals 1, no sampling is done on adj matrix.")
# parser.add_argument("--baseblock", default="res", help="The base building block (resgcn, densegcn, mutigcn, inceptiongcn).")
parser.add_argument("--nbaseblocklayer", type=int, default=1,
                    help="The number of layers in each baseblock")
parser.add_argument("--aggrmethod", default="default",
                    help="The aggrmethod for the layer aggreation. The options includes add and concat. Only valid in resgcn, densegcn and inecptiongcn")
parser.add_argument("--task_type", default="semi", help="The node classification task type (full and semi). Only valid for cora, citeseer and pubmed dataset.")

args = parser.parse_args()


if args.ssl == 'AttributeMask':
    args.pca = 1

# if args.unsupervised_mode:
#     from ssl_model import *
# else:
#     from models import *
from models import *

if args.debug:
    print(args)

if args.write_json:
    args.no_tensorboard = True

if args.lambda_ != 0 and args.ssl != 'Base' and not args.param_searching:
    from configs import *
    args.lambda_ = lambda_config5[args.ssl][args.dataset]

# pre setting
args.cuda = not args.no_cuda and torch.cuda.is_available()

args.mixmode = args.no_cuda and args.mixmode and torch.cuda.is_available()

if args.aggrmethod == "default":
    if args.type == "resgcn":
        args.aggrmethod = "add"
    else:
        args.aggrmethod = "concat"

if args.unsupervised_mode and args.early_stopping > 0:
    args.early_stopping = 0
    print("In the unsupervised mode, early_stopping is not valid option. Setting early_stopping = 0.")

if args.fastmode and args.early_stopping > 0:
    args.early_stopping = 0
    print("In the fast mode, early_stopping is not valid option. Setting early_stopping = 0.")
if args.type == "mutigcn":
    print("For the multi-layer gcn model, the aggrmethod is fixed to nores and nhiddenlayers = 1.")
    args.nhiddenlayer = 1
    args.aggrmethod = "nores"

# random seed setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda or args.mixmode:
    torch.cuda.manual_seed(args.seed)

# should we need fix random seed here?
sampler = Sampler(args.dataset, args, args.datapath, args.task_type)

# get labels and indexes
labels, idx_train, idx_val, idx_test = sampler.get_label_and_idxes(args.cuda)
nfeat = sampler.nfeat
nclass = sampler.nclass
print("nclass: %d\tnfea:%d" % (nclass, nfeat))

hidden = torch.load('{0}-Base-{1}-pretrained.pt'.format(args.dataset, args.hidden)).numpy()
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
A_feat = kneighbors_graph(sampler.features, 5, mode='connectivity', include_self=False)
A_hidden = kneighbors_graph(hidden, 5, mode='connectivity', include_self=False)

import ipdb
ipdb.set_trace()

# A_feat = radius_neighbors_graph(sampler.features, 5, mode='connectivity', include_self=False)
# A_hidden = radius_neighbors_graph(hidden, 5, mode='connectivity', include_self=False)

def overlap(A, B):
    pass


if __name__ == "__main__":
    main()
