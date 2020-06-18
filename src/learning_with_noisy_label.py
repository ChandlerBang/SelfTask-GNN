from __future__ import division
from __future__ import print_function

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
parser.add_argument('--hidden', type=int, default=128,
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
parser.add_argument("--task_type", default="full", help="The node classification task type (full and semi). Only valid for cora, citeseer and pubmed dataset.")

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
    args.lambda_ = lambda_config4[args.ssl][args.dataset]

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

# The model
model = GCNModel(nfeat=nfeat,
                 nhid=args.hidden,
                 nclass=nclass,
                 nhidlayer=args.nhiddenlayer,
                 dropout=args.dropout,
                 baseblock=args.type,
                 inputlayer=args.inputlayer,
                 outputlayer=args.outputlayer,
                 nbaselayer=args.nbaseblocklayer,
                 activation=F.relu,
                 withbn=args.withbn,
                 withloop=args.withloop,
                 aggrmethod=args.aggrmethod,
                 mixmode=args.mixmode)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.618)
if args.steplr:
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 500, 1000, 2000, 4000, 8000], gamma=0.5)
else:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000], gamma=1)
# convert to cuda

if args.finetune:
    model.load_state_dict(torch.load('{0}-{1}-initialization.pt'.format(args.dataset, args.ssl)))

if args.cuda:
    model.cuda()
if args.ssl is None or args.lambda_ == 0:
    args.ssl = 'Base'

# For the mix mode, lables and indexes are in cuda.
if args.cuda or args.mixmode:
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


if args.warm_start is not None and args.warm_start != "":
    early_stopping = EarlyStopping(fname=args.warm_start, verbose=False)
    print("Restore checkpoint from %s" % (early_stopping.fname))
    model.load_state_dict(early_stopping.load_checkpoint())

# set early_stopping
if args.early_stopping > 0:
    early_stopping = EarlyStopping(patience=args.early_stopping, verbose=False)
    print("Model is saving to: %s" % (early_stopping.fname))

if args.no_tensorboard is False and not args.unsupervised_mode:
    # tb_writer = SummaryWriter(
    #     comment=f"-dataset_{args.dataset}-type_{args.ssl}-label{args.label_rate}-lambda{args.lambda_}")
    if args.fastmode:
        dirpath = f"./runs/no_val/{args.dataset}-{args.ssl}-label{args.label_rate}-lambda{args.lambda_}-pretrain{args.load_pretrain}"
    else:
        dirpath = f"./runs/with_val/{args.dataset}-{args.ssl}-label{args.label_rate}-lambda{args.lambda_}-pretrain{args.load_pretrain}"
    if osp.exists(dirpath):
        shutil.rmtree(dirpath)
    tb_writer = SummaryWriter(logdir=dirpath)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# define the training function.
def train(epoch, train_adj, train_fea, idx_train, val_adj=None, val_fea=None):

    if val_adj is None:
        val_adj = train_adj
        val_fea = train_fea

    t = time.time()
    model.train()
    optimizer.zero_grad()

    # if epoch % 1 == 0 and args.ssl != 'Base':
    #     ssl_agent.reset()

    if args.ssl == 'NeighborMask':
        output, embeddings = ssl_agent.get_model_output(model)
    else:
        output, embeddings = model.myforward(train_fea, train_adj)

    # special for reddit
    if sampler.learning_type == "inductive":
        if args.selftraining:
            loss_train = F.nll_loss(output, sampler.labels_st[idx_train])
            acc_train = accuracy(output, sampler.labels_st[idx_train])
        else:
            loss_train = F.nll_loss(output, labels[idx_train])
            acc_train = accuracy(output, labels[idx_train])
    else:
        if args.selftraining:
            loss_train = F.nll_loss(output[idx_train], sampler.labels_st[idx_train])
            acc_train = accuracy(output[idx_train], sampler.labels_st[idx_train])
        else:
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])


    if epoch >= 0:
        ssl_agent.noisy_label = True

    loss_ssl = args.lambda_ * ssl_agent.make_loss(embeddings)

    if args.unsupervised_mode:
        loss_total = loss_ssl
    else:
        loss_total = loss_train + loss_ssl
        # loss_total = loss_train


    # corrected = None
    # if epoch > 50:
    #     idx_unlabeled = np.array([x for x in range(len(labels)) if x not in idx_train])
    #     from ssl_utils import label_correction
    #     # labels_noisy = ssl_agent.agent.concated
    #     # idx_selected = (ssl_agent.agent.probs.max(1) > 0.3)

    #     labels_noisy = np.load(f'preds/{args.dataset}_{args.seed}_pred.npy').argmax(1)
    #     idx_selected = (np.load(f'preds/{args.dataset}_{args.seed}_pred.npy').max(1) > 0.3)

    #     idx_selected_ = np.zeros(idx_selected.shape)
    #     idx_selected_[idx_train.cpu().numpy()] = -1
    #     idx_selected_ = (idx_selected_[idx_selected]!=-1)
    #     print('len(idx_selected) = %s' % idx_selected.sum())

    #     acc = ((labels.cpu().numpy() == labels_noisy)[idx_selected][idx_selected_].sum())/ idx_selected.sum()
    #     print('corrected acc: %s' % acc)
    #     if corrected is None:
    #         corrected = np.copy(labels_noisy[idx_selected])
    #     corrected = label_correction(output.detach()[idx_selected], corrected, idx_train)
    #     acc = ((labels[idx_selected] == corrected)[idx_selected_].sum()).item()/ len(corrected)
    #     print('corrected acc: %s' % acc)
    #     loss_total += 0.1*F.nll_loss(output[idx_selected], corrected)

    # loss_total += 0.1*F.nll_loss(output[sampler._idx_train], sampler.labels_st[sampler._idx_train])

    loss_total.backward()
    optimizer.step()
    train_t = time.time() - t
    val_t = time.time()
    # We can not apply the fastmode for the reddit dataset.
    # if sampler.learning_type == "inductive" or not args.fastmode:

    # if args.early_stopping > 0 and sampler.dataset != "reddit":
    #     loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
    #     early_stopping(loss_val, model)

    if not args.fastmode and args.early_stopping > 0:
        #    # Evaluate validation set performance separately,
        #    # deactivates dropout during validation run.
        model.eval()
        output = model(val_fea, val_adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
        acc_val = accuracy(output[idx_val], labels[idx_val]).item()
        early_stopping(acc_val, model)
        # early_stopping(-loss_val, model)
    else:
        loss_val = 0
        acc_val = 0

    if args.lradjust:
        scheduler.step()

    val_t = time.time() - val_t

    # from utils import get_pairwise_sim
    # labelled_sim = get_pairwise_sim(embeddings[idx_train])
    # unlabelled_sim = get_pairwise_sim(embeddings[np.union1d(idx_test, idx_val)])
    # all_sim = get_pairwise_sim(embeddings)
    # print("Labeled Node Similarity: {:.4f}".format(labelled_sim.item()), "Unlabeled Node Similarity: {:.4f}".format(unlabelled_sim.item()), 'Total Similarity: {:.4f}'.format(all_sim.item()))

    # return (loss_train.item(), acc_train.item(), loss_val, acc_val, get_lr(optimizer), train_t, val_t)
    try:
        return (loss_train.item(), acc_train.item(), loss_val, acc_val, loss_ssl.item(), loss_total.item(), train_t)
    except:
        return (loss_train.item(), acc_train.item(), loss_val, acc_val, loss_ssl, loss_total.item(), train_t)

def test(test_adj, test_fea):
    model.eval()
    # output = model(test_fea, test_adj)
    output, embeddings = model.myforward(test_fea, test_adj)

    # import ipdb
    # ipdb.set_trace()
    # loss_ssl = ssl_agent.make_loss(embeddings)
    # print(loss_ssl)

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    auc_test = roc_auc_compute_fn(output[idx_test], labels[idx_test])

    if False:
    # if not args.write_json:
        from utils import get_pairwise_sim
        labelled_sim = get_pairwise_sim(embeddings[idx_train]).item()

        unlabelled_sim = get_pairwise_sim(embeddings[np.union1d(idx_test.cpu(), idx_val.cpu())]).item()
        all_sim=labelled_sim
        # all_sim = get_pairwise_sim(embeddings)
        print("Labeled Node Similarity: {:.4f}".format(labelled_sim.item()), "Unlabeled Node Similarity: {:.4f}".format(unlabelled_sim.item()), 'Total Similarity: {:.4f}'.format(all_sim.item()))
    else:
        labelled_sim = 0
        unlabelled_sim = 0
        all_sim = 0

    if args.finetune:
        args.load_pretrain = 3
    if args.pca:
        args.ssl += '~PCA'

    if args.write_res:
        np.save(f'preds/{args.dataset}_{args.seed}_pred.npy', output.detach().cpu().numpy())

    if args.write_json:
        import json
        nlayers = args.nhiddenlayer * args.nbaseblocklayer + 2
        res = {'labelled_sim': labelled_sim, 'unlabelled_sim': unlabelled_sim, 'all_sim': all_sim, 'test_loss': loss_test.item(), 'test_acc': acc_test.item(), 'loss_train': loss_train[epoch], 'loss_ssl': loss_ssl[epoch]}
        # with open('results_ssl_norm_glorot/{0}_{1}-{2}_{3}_{4}.json'.format(args.dataset, args.type, args.ssl, nlayers, args.seed), 'w') as f:
        # with open('results-pretrain/{0}_{1}-{2}_{3}_{4}.json'.format(args.dataset, args.label_rate, args.ssl, args.load_pretrain, args.seed), 'w') as f:
        # with open('results-nbm/{0}_{1}-{2}_{3}_{4}.json'.format(args.dataset, args.label_rate, args.ssl, args.load_pretrain, args.seed), 'w') as f:
        # with open('results-baseline/{0}_{1}-{2}_{3}_{4}.json'.format(args.dataset, args.label_rate, args.ssl, args.load_pretrain, args.seed), 'w') as f:
        # with open('results-pca/{0}_{1}-{2}_{3}_{4}.json'.format(args.dataset, args.label_rate, args.ssl, args.load_pretrain, args.seed), 'w') as f:
        # with open('results-val/{0}_{1}-{2}_{3}_{4}.json'.format(args.dataset, args.label_rate, args.ssl, args.load_pretrain, args.seed), 'w') as f:
        # with open('results-new-seeds/{0}_{1}-{2}_{3}_{4}.json'.format(args.dataset, args.label_rate, args.ssl, args.load_pretrain, args.seed), 'w') as f:
        # with open('results-3layers/validation/{0}_{1}-{2}_{3}_{4}.json'.format(args.dataset, args.label_rate, args.ssl, args.load_pretrain, args.seed), 'w') as f:
        with open('results-3layers/train_size/{0}_{1}-{2}_{3}_{4}.json'.format(args.dataset, args.train_size, args.ssl, args.load_pretrain, args.seed), 'w') as f:
            json.dump(res, f)

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "auc= {:.4f}".format(auc_test),
          "accuracy= {:.4f}".format(acc_test.item()))
    print("accuracy=%.5f" % (acc_test.item()))
    return (loss_test.item(), acc_test.item())


# Train model
t_total = time.time()
loss_train = np.zeros((args.epochs,))
acc_train = np.zeros((args.epochs,))
loss_val = np.zeros((args.epochs,))
acc_val = np.zeros((args.epochs,))
loss_ssl = np.zeros((args.epochs,))

sampling_t = 0


#####################################
from selfsl import *

nclass = max(labels).item() + 1

if args.finetune:
    ssl_agent = Base(sampler.adj, sampler.features, device='cuda')
    optimizer = optim.Adam([y for x,y  in model.named_parameters() if 'ingc' not in x],
                           lr=args.lr, weight_decay=args.weight_decay)

if args.load_pretrain:
    ssl_agent = Base(sampler.adj, sampler.features, device='cuda')

if args.ssl is None or args.lambda_ == 0 or args.ssl == 'Base':
    ssl_agent = Base(sampler.adj, sampler.features, device='cuda')
    args.ssl = 'Base'

if args.ssl == 'EdgeMask':
    ssl_agent = EdgeMask(sampler.adj, sampler.features, device='cuda')

if args.ssl == 'EdgeMelt':
    ssl_agent = EdgeMelt(sampler.adj, sampler.features, device='cuda')

if args.ssl == 'DistanceCluster':
    ssl_agent = DistanceCluster(sampler.adj, sampler.features, idx_train=idx_train, nhid=args.hidden, device='cuda')
    optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)

if args.ssl == 'SpectralPredictor':
    ssl_agent = SpectralPredictor(sampler.adj, sampler.features, nhid=args.hidden, args=args, device='cuda')
    optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)

if args.ssl == 'PairwiseDistance':
    ssl_agent = PairwiseDistance(sampler.adj, sampler.features, idx_train=idx_train, nhid=args.hidden, device='cuda')
    optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)

if args.ssl == 'LabeledDistance':
    ssl_agent = LabeledDistance(sampler.adj, sampler.features, sampler.labels, nclass=nclass, idx_train=idx_train, nhid=args.hidden, device='cuda')
    optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)

if args.ssl == 'ICAPseudoLabel':
    ssl_agent = ICAPseudoLabel(sampler.adj, sampler.features, sampler.labels, nclass=nclass, idx_train=idx_train, nhid=args.hidden, device='cuda', args=args)
    optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)

if args.ssl == 'LPPseudoLabel':
    ssl_agent = LPPseudoLabel(sampler.adj, sampler.features, sampler.labels, nclass=nclass, idx_train=idx_train, nhid=args.hidden, device='cuda', args=args)
    optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)

if args.ssl == 'ContextPredictor':
    ssl_agent = ContextPredictor(sampler.adj, sampler.features, device='cuda')

if args.ssl == 'NeighborMask':
    ssl_agent = NeighborMask(sampler.adj, sampler.features, idx_train=idx_train, nhid=args.hidden, device='cuda')
    optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)

if args.ssl == 'AttributeMask':
    ssl_agent = AttributeMask(sampler.adj, sampler.features, idx_train=idx_train, nhid=args.hidden, device='cuda')
    optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)
if args.ssl == 'NodeProperty':
    ssl_agent = NodeProperty(sampler.adj, sampler.features, nhid=args.hidden, device='cuda')
    optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)


if args.ssl == 'DeepClulster':
    ssl_agent = DeepClulster(sampler.adj, sampler.features, nhid=args.hidden, nclusters=20, device='cuda')
    optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)


for epoch in range(args.epochs):
    input_idx_train = idx_train
    sampling_t = time.time()
    # no sampling
    # randomedge sampling if args.sampling_percent >= 1.0, it behaves the same as stub_sampler.

    train_adj, train_fea = ssl_agent.transform_data()
    if args.identity:
        train_adj = torch.eye(sampler.adj.shape[0]).to(device)

    # (train_adj, train_fea) = sampler.randomedge_sampler(percent=args.sampling_percent, normalization=args.normalization, cuda=args.cuda)

    if args.mixmode:
        train_adj = train_adj.cuda()

    sampling_t = time.time() - sampling_t

    # The validation set is controlled by idx_val
    # if sampler.learning_type == "transductive":
    if False:
        outputs = train(epoch, train_adj, train_fea, input_idx_train)
    else:
        (val_adj, val_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
        # val_adj, val_fea = ssl_agent.transform_data()

        if args.identity:
            val_adj = torch.eye(sampler.adj.shape[0]).to(device)

        if args.mixmode:
            val_adj = val_adj.cuda()
        outputs = train(epoch, train_adj, train_fea, input_idx_train, val_adj, val_fea)

    if args.debug and epoch % 1 == 0:
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(outputs[0]),
              'acc_train: {:.4f}'.format(outputs[1]),
              'loss_val: {:.4f}'.format(outputs[2]),
              'acc_val: {:.4f}'.format(outputs[3]),
              'loss_ssl: {:.4f}'.format(outputs[4]),
              'loss_total: {:.4f}'.format(outputs[5]),
              't_time: {:.4f}s'.format(outputs[6]))

    if args.no_tensorboard is False and not args.unsupervised_mode:
        tb_writer.add_scalars('Loss', {'class': outputs[0], 'ssl': outputs[4] , 'total': outputs[5], 'val': outputs[2]}, epoch)
        tb_writer.add_scalars('Accuracy', {'train': outputs[1], 'val': outputs[3]}, epoch)
        # tb_writer.add_scalar('lr', outputs[4], epoch)
        # tb_writer.add_scalars('Time', {'train': outputs[5], 'val': outputs[6]}, epoch)


    loss_train[epoch], acc_train[epoch], loss_val[epoch], acc_val[epoch], loss_ssl[epoch] = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]

    if args.early_stopping > 0 and early_stopping.early_stop:
        print("Early stopping.")
        model.load_state_dict(early_stopping.load_checkpoint())
        break

if args.early_stopping > 0:
    model.load_state_dict(early_stopping.load_checkpoint())
    # print('=== best score: %s, epoch %s ===' % (early_stopping.best_score, early_stopping.best_epoch))
    print('=== best score: %s, loss_val: %s, epoch %s ===' % (early_stopping.best_score, loss_val[early_stopping.best_epoch], early_stopping.best_epoch))
    print('For this epoch, val loss: %s, val acc: %s' % (loss_val[early_stopping.best_epoch], acc_val[early_stopping.best_epoch]))

if args.debug:
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
(test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)

if args.identity:
    test_adj = torch.eye(sampler.adj.shape[0]).to(device)

if args.mixmode:
    test_adj = test_adj.cuda()

(loss_test, acc_test) = test(test_adj, test_fea)
print("%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f" % (
loss_train[-1], loss_val[-1], loss_test, acc_train[-1], acc_val[-1], acc_test))
print('Self-Supervised Type: %s' % args.ssl)

if args.unsupervised_mode:
    torch.save(model.state_dict(), '{0}-{1}-initialization.pt'.format(args.dataset, args.ssl))

assert False, 'Program Stop Here'
if args.selftraining:
    assert False, 'Program Stop Here'

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
def lr_test(test_adj, test_fea):
    model.eval()
    output, embeddings = model.myforward(test_fea, test_adj)

    embeddings = embeddings.detach()

    if args.unsupervised_mode:
        torch.save(embeddings, '{0}-{1}-pretrained.pt'.format(args.dataset, args.ssl))

    y_train, y_test = labels[idx_train], labels[idx_test]
    y_train, y_test = y_train.cpu().numpy(), y_test.cpu().numpy()
    X_train, X_test = embeddings[idx_train], embeddings[idx_test]
    X_train, X_test = X_train.cpu().numpy(), X_test.cpu().numpy()

    if args.fastmode:
        best_C = param_search(X_train, y_train)
    else:
        X_val = embeddings[idx_val].cpu().numpy()
        y_val = labels[idx_val].cpu().numpy()
        best_C = param_search(np.vstack((X_train, X_val)), np.hstack((y_train, y_val)))

    from sklearn.svm import SVC, LinearSVC
    # lr = LinearSVC(C=1)
    lr = LogisticRegression(penalty='l2', C=best_C, random_state=1, max_iter=10000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print('=== LR Test Accuracy: %s' % metrics.accuracy_score(y_pred, y_test), 'feature dim =', X_train.shape[1])

test_fold = torch.ones(sampler.adj.shape[0])
test_fold[idx_train] = -1
test_fold[idx_val] = 0
test_fold = test_fold[test_fold != 1].cpu().numpy()

def param_search(X, y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import PredefinedSplit
    ps = PredefinedSplit(test_fold=test_fold)
    C = [1e-4, 0.001, 0.01, 0.1, 1, 10, 50, 100, 200, 500, 1000]
    param_grid = dict(C=C)
    lr = LogisticRegression(penalty='l2', random_state=1, max_iter=10000)
    if args.fastmode:
        grid_search = GridSearchCV(lr, param_grid, scoring="accuracy", n_jobs=-1, cv=4)
    else:
        grid_search = GridSearchCV(lr, param_grid, scoring="accuracy", n_jobs=-1, cv=ps)
    grid_result = grid_search.fit(X, y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return grid_result.best_params_['C']


def lr_baseline():
    embeddings = sampler.features
    y_train, y_test = labels[idx_train], labels[idx_test]
    y_train, y_test = y_train.cpu().numpy(), y_test.cpu().numpy()
    X_train, X_test = embeddings[idx_train], embeddings[idx_test]
    X_train, X_test = X_train.cpu().numpy(), X_test.cpu().numpy()

    if args.fastmode:
        best_C = param_search(X_train, y_train)
    else:
        X_val = embeddings[idx_val].cpu().numpy()
        y_val = labels[idx_val].cpu().numpy()
        best_C = param_search(np.vstack((X_train, X_val)), np.hstack((y_train, y_val)))

    lr = LogisticRegression(penalty='l2', C=best_C, random_state=1)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print('=== LR Baseline Test Accuracy: %s' % metrics.accuracy_score(y_pred, y_test), 'feature dim =', X_train.shape[1])

if not args.write_json:
    lr_test(test_adj, test_fea)
    lr_baseline()

print(args)
nnodes = sampler.adj.shape[0]
print('len(idx_train)/len(adj.shape[0])= ',len(idx_train)/nnodes)

if not args.unsupervised_mode:
    tb_writer.close()

