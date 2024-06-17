#=====================================================================================
"""
Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
#=====================================================================================

from pathlib import Path
import math
import sys
import argparse
import os
from glob import glob
import itertools
import random
import pickle
import timeit
import time
from collections import OrderedDict


from scipy.optimize import least_squares
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from ase.units import mol, kcal
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from pykinml import data
from pykinml import nnpes


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



# ====================================================================================================
# ====================================================================================================
def parse_arguments_list():
    parser = argparse.ArgumentParser(description='PES code')
    
    #Data arguments
    parser.add_argument('-if', '--fidlevel', type=int, default=1,
                        help='Specify input fidelity level (SQLite db only)')
    parser.add_argument('--fidlevel-lf', type=int, default=None,
                        help='Specify input low fidelity level as integer [0, 1, 2, 3, 4] (SQLite db only)')
    parser.add_argument('-d', '--input-data-type', choices=('aev', 'pca', 'sqlite'), default='sqlite',
                        help='specify input data file type [sqlite]')
    parser.add_argument('-f', '--input-data-fname', nargs='*', default=['data_holder/C5H5/C*H*.db'],
                        help='Specify input data filename(s)')
    parser.add_argument('--trtsid-name', nargs='*', default=['sample_tvt.txt'],
                        help='Specify training(and validation)/test input data xyzid with tvt mask [sample_tvt.txt]')
    parser.add_argument('--tvt', nargs='+', type=float, default=[0.8, 0.1, 0.1],
                        help='Specify set of the decimal fraction (0.0,1.0] or number of points for training, validation and test set. The sum should be 1 if you input decimal points')
    parser.add_argument('--present_elements', nargs='+', default=['C', 'H'],
                        help='Specify how many chemical elements are included in the training set')
    parser.add_argument('--temp', type=int, default=None,
                        help='Specify temperature that QC data was sampled. default = None')
    parser.add_argument('-sae', '--sae-fit', action='store_true', default=True,
                        help='compute single atom energies and subtract them from the dataset')
    parser.add_argument('-ns', '--nameset', nargs='+', default=None,
                        help='Specify set of the patterns of db name to generate redueced set. [w irc ...]')
    parser.add_argument('-prs', '--pre-saved', action='store_true', default=False,
                        help='indicates that the data has already been preped and saved in the proper location')
    parser.add_argument('--data_path', type=str, default='',
                        help='if pre_saved, the path to the saved data')
    #Model Arguments
    parser.add_argument('-o', '--optimizer', nargs='*', default=['Adam'],
                        help='Specify the optimizer type [SGD or Adam or AdamW]')
    parser.add_argument('-ls', '--lrscheduler', nargs='*', default='rop',
                        help='Specify learning rate scheduler. [exp, step, rop]')
    parser.add_argument('-l', '--learning-rate', type=float, default=1.e-3,
                        help='specify optimizer learning rate [1.e-3]')
    parser.add_argument('-dr', '--decayrate', type=float, default=0.5,
                        help='Specify learning rate scheduler decay rate. default = 0.1')
    parser.add_argument('-lrp', '--LR_patience', type=float, default=25,
                        help='patience for rop lr scheduler. default = 25')
    parser.add_argument('-lrt', '--LR_threshold', type=float, default=0.05,
                        help='validation error threshold for rop lr scheduler. default = 0.05 kcal/mol')
    parser.add_argument('-m', '--momentum', type=float, default=0.5, help='specify optimizer momentum [0.5]')
    parser.add_argument('--weight-decay', type=float, default=0, help='specify L2-penalty for regularization')
    parser.add_argument('-e', '--epochs', type=int, default=2, help='specify number of epochs for training')
    parser.add_argument('--save_every', type=int, default=10,
                        help='specify save NN every <save_every> epochs')
    
    parser.add_argument('-lmo', '--load-model', action='store_true', default=False,
                        help='specify new model, do not load exising net0.pt, net1.pt, opt.pt')
    parser.add_argument('-lm', '--load_model_name', nargs='*', default=['comp.pt'],
                        help='Specify load model file name [Comp.pt] or the path of folders which contains model parameters [net_pars/]')

    parser.add_argument('-btr', '--tr_batch_size', type=int, default=10,
                        help='specify (approximate) batch size for training')
    parser.add_argument('-bvl','--vl_batch_size', type=int,  default=10,                            #Use to avoid memory issues when doing force training.
                        help='validation set batch size')
    parser.add_argument('-bts', '--ts_batch_size', type=int,  default=10,                            #Use to avoid memory issues when doing force training.
                        help='test set batch size')


    parser.add_argument('-nl', '--my-neurons', nargs='+', type=int, default=[128, 64, 32, 1],
                        help='Specify set of the number of neurons in each layer. The last number is for the output layer (new model only)')
    parser.add_argument('-al', '--my-actfn', nargs='+', default=["gaussian", "gaussian", "gaussian", "identity"],
                        help='Specify set of the activation functions in each layer. [gaussian, tanh, identity, relu, silu] (new model only)')
    parser.add_argument('-sn', '--savenm', nargs='*', default='my_model',
                        help='Specify folder name to save the data. The optimizer and device will be appended to the name.')
    parser.add_argument('-r', '--randomseed', nargs='+', type=int, default=[0, 1], help='Specify random seed as integer')
    parser.add_argument('--fw', type=float, default=0.0,
                        help='Specify the decimal fraction (0.0,1.0] for weighted loss of force. Only available when floss option is on')
    parser.add_argument('--floss', action='store_true', default=False, help='Use force in loss function')
    parser.add_argument('--write-tvtmsk', action='store_true', default=False,
                        help='write training/validation/test mask')
    parser.add_argument('--read-trid', action='store_true', default=True,
                        help='read training set xyzid list')
    parser.add_argument('-ofw', '--optimize-force-weight', action='store_true', default=False,
                        help='optimize relative weights of energy and force in loss function')
    parser.add_argument('--no-biases', action='store_true', default=False,
                        help='dont include biases in NN')
    parser.add_argument('--delta', action='store_true', default=False,
                        help='delta learning to predict diference between 2 fidelity levels')
    parser.add_argument('--multi_fid', action='store_true', default=False,
                        help='Train to multiple fidelity levels')

    #Device arguments
    parser.add_argument('--ddp', action='store_true',  default=False,
                        help='Use pytorches Distributed Data Parallel')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0],
                        help='GPU ids to use for DDP') 
    parser.add_argument('-c', '--num_cores', type=int, default=4,
                        help='specify number of cores to run on [4], nb. num_threads=2 x num_cores')
    
    #AEV arguments
    parser.add_argument('--beta', type=float, default=0.95,
                        help='Scaling factor for angular component of SAE')
    parser.add_argument('-R_c', '--cuttoff-radius', type=float, nargs='+',  default=[5.2, 3.8],
                        help='radial and angular cuttoff radii')
    parser.add_argument('-aev', '--aev_params', type=int, nargs='+',  default=[16, 8, 8],
                        help='parameters determining the length of the AEV')

    args = parser.parse_args()



    return args



def kcpm(E_eV):
    """
    converts energy in eV to kcal/mol
    """
    return E_eV * mol / kcal


def my_loss(ediff, fdiff = [], dEsq=1., dfsq=1., p=2):
    """
    loss function used during training
    Inputs:
        ediff: absolute difference between predicted and target energies.
        fdiff: absolute difference between predicted and target forces. Empty if not performing force training.
        dEsq: scaling factor for energy loss.
        dfsq: scaling factor for force loss
        p: power to raise total loss. E.g. p=1 return L1 loss, p=2 returns L2 loss.
    
    Output:
        ls_e (float): energy component of the loss.
        ls_f (float): force component of the loss. 0 if not training to forces.
    """
    
    ls_f = 0.
    ls_e = torch.mean(ediff ** p)/dEsq
    for i in range(len(fdiff)):
        ls_f += torch.mean(fdiff[i] ** p)
    ls_f = ls_f/len(ediff)/dfsq
    return ls_e, ls_f

def force_abs_dif(pred_forces, true_forces, fdims):
    fdiff = []
    for i in range(len(pred_forces)):
        fdiff.append(abs(pred_forces[i][0:fdims[i]] - true_forces[i][0:fdims[i]]))
    return fdiff

def energy_abs_dif(pred_engs, true_engs):
    ediff = abs(pred_engs - true_engs)
    return ediff

class task_weights(torch.nn.Module):
    def __init__(self, num_tasks=2, log_sigma=None, traditional=False):
        super(task_weights, self).__init__()
        """
        class to emulate the work done in this paper: 
        Kendall, Alex, Yarin Gal, and Roberto Cipolla. 
        "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics." 
        Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018

        Also includes a "traditional" method to be used if you don't want to optimizes the 
        task weights and prefer to leave them fixed. traditional method can be used with
        task weight optimization but runs the risk of the model "forgeting" one of the loss terms.


        loss_terms:  list of loss values, one for each task to be learned, e.g. Energy, Force, Dipole, etc.
        log_sigmas:  parameter used to determine weight for each loss term. If args.ofw is True they are optimized 
                     by the network. Otherwise they are static.
        Traditional: if Traditional is True, will use the formula Total_loss=(1-a)*Energy_loss+a*Force_loss.
                     Assumes that Energy_loss is loss_term[0]
        """
        self.num_tasks = num_tasks
        if traditional:
            if log_sigma == None:
                log_sigma = torch.ones(num_tasks-1, requires_grad=True)/num_tasks
        else:
            if log_sigma == None:
                log_sigma = torch.zeros((num_tasks), requires_grad=True)
        self.log_sigma = log_sigma
        self.traditional = traditional

    def get_precisions(self, log_sigma):
        """
        A factor of 0.5 has been factored out and ignored as it scales all terms equally.
        """
        return 1 / torch.exp(log_sigma)
        #return 1 / self.log_sigma**2

    def forward(self, loss_terms, log_sigma):
        """
        When using traditional weighting and task weight optimization the task weights should be clamped to between 0.001 and 0.999
        to prevent negative contributions to the loss. This problem cn occur with non traditional weighting as well. Looking for way around it...
        """
        #self.log_sigma=log_sigma
        total_loss = 0
        if self.traditional:
            #total_loss += torch.clamp(1-torch.sum(self.log_sigma), 0.001, 0.999) * loss_terms[0]
            total_loss += (1-torch.sum(log_sigma)) * loss_terms[0]
            for task in range (1, self.num_tasks):
                #total_loss += torch.clamp(self.log_sigma[task-1], 0.001, 0.999) * loss_terms[task]
                total_loss += log_sigma[task-1] * loss_terms[task]
        else:
            self.precisions = self.get_precisions(log_sigma)
            for task in range(self.num_tasks):
                total_loss += self.precisions[task] * loss_terms[task] + log_sigma[task]
        return total_loss


def set_up_task_weights(model, args, optimizer):
    if not args.floss:
        args.optimize_force_weight = False
        args.fw = 0.0
    if args.optimize_force_weight:
        log_sigma = torch.zeros((2), requires_grad=True)
        mtl = task_weights(num_tasks=2, traditional=False)#.to(self.gpu_id)
        model.log_sigma = torch.nn.Parameter(log_sigma)
        optimizer.param_groups[0]['params'].append(model.log_sigma)
        print('Force weight optimization is ON!')
    else:
        log_sigma = torch.tensor([args.fw], requires_grad=False)
        mtl = task_weights(num_tasks=2, traditional=True)#.to(self.gpu_id)
        model.log_sigma = torch.nn.Parameter(log_sigma)
    model.mtl = mtl


def load_trained_model(args, load_opt=True):
    checkpoint = torch.load(args.load_model_name)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        if k[:7] == 'module.':
            name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
        else:
            name = k
        new_state_dict[name] = v
    #log_sigma = checkpoint['model_state_dict'].pop('log_sigma', None)
    args.netparams = checkpoint['params']
    args.my_actfn = args.netparams['activations'][0]
    prep_netarch(args)
    args.netparams['activations'] = [args.my_activations for i in range(args.num_species)]
    if args.multi_fid:
        net = nnpes.CompositeNetworks_MF(**args.netparams)
    else:
        net = nnpes.CompositeNetworks(**args.netparams)
    net.load_state_dict(new_state_dict, strict=False)
    if load_opt:
        args.weight_decay = checkpoint['optimizer_state_dict']['param_groups'][0]['weight_decay']
        args.learning_rate = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
        my_lr_scheduler = checkpoint['lr_scheduler']
        print(' learning rate loaded')
        return net, my_lr_scheduler
    else:
        return net


def prep_model(args, load_opt=True,
               num_nn=None, din_nn=256,
               my_neurons_a=None,
               my_neurons_b=None,
               sae_energies=np.zeros(2), biases=True):


    print('model random seed:', args.randomseed[1])
    random.seed(args.randomseed[1])
    torch.manual_seed(random.randrange(200000))
    np.random.seed(random.randrange(200000))
    random.seed(random.randrange(200000))
    if args.load_model:
        if load_opt:
            net, my_lr_scheduler = load_trained_model(args, load_opt)
        else:
            net = load_trained_model(args, load_opt)
    else:
        prep_netarch(args)
        acts = [args.my_activations for i in range(args.num_species)]
        neus = [my_neurons_a for i in range(args.num_species)]
        dins = [din_nn for i in range(args.num_species)]
        if args.multi_fid:
            args.netparams = {
                             'n_input': dins,
                             'neurons': neus, 'activations': acts,
                             'neurons_hf': neus, 'activations_hf': acts,
                             'sae_energies': sae_energies, 'add_sae': True, 'biases': biases,
                             'pass_eng': True, 'pass_aev': True
                             }
            net = nnpes.CompositeNetworks_MF(**args.netparams)
        else:
            activations = [args.my_activations for i in range(args.num_species)]
            args.netparams = {
                             'n_input': dins,
                             'neurons': neus, 'activations': acts,
                             'sae_energies': sae_energies, 'add_sae': True, 'biases': biases
                             }
            net = nnpes.CompositeNetworks(**args.netparams)

    args.netparams['activations'] = [args.my_actfn for i in range(args.num_species)]

    net.netparams = args.netparams
    net.sae_energies = sae_energies
    if args.optimizer[0] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum,
                              nesterov=True, weight_decay=args.weight_decay)
    elif args.optimizer[0] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer[0] == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    #optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if not load_opt or not args.load_model:
        lrscheduler = args.lrscheduler#[0]
        print('lrscheduler: ', args.lrscheduler)
        if lrscheduler == 'exp':
            my_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.decayrate)
        elif lrscheduler == 'step':
            my_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.decayrate, last_epoch=-1)
        elif lrscheduler == 'rop':
            if args.tvt[1] == 0:
                print('ReduceLROnPlateau LR schecular requires validation set')
                sys.exit()
            my_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decayrate, patience=args.LR_patience, threshold=args.LR_threshold, verbose=True)


    print('net:', net)
    
    #for name, param in net.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.data)

    return net, optimizer, my_lr_scheduler




def args_to_actfn(argsact):
    actfn = []
    if isinstance(argsact, list):
        for i in range(0, len(argsact)):
            if argsact[i] == 'gaussian':
                actfn.append(nnpes.gaussian)
            elif argsact[i] == 'relu':
                actfn.append(nnpes.my_relu)
            elif argsact[i] == 'silu':
                actfn.append(nnpes.silu)
            elif argsact[i] == 'tanh':
                actfn.append(nn.Tanh())
            elif argsact[i] == 'identity':
                actfn.append(nnpes.identity)
    else:
        for i in range(0, argsact):
            actfn.append(nnpes.gaussian)
        actfn[-1] = nnpes.identity

    return actfn



def prep_netarch(args):
    args.my_activations = args_to_actfn(args.my_actfn)
    print('activation functions:', args.my_actfn)
    return 0



def read_xidtxt(fname, tvtmsk=False, std=False):
    try:
        f = open(fname, "r")
    except IOError:
        print("Could not open file:" + fname)
        sys.exit()
    with f:
        xyz = f.readlines()

    xid = []
    for line in range(0, len(xyz)):
        if tvtmsk:
            if std:
                lst = (" ".join(xyz[line].split())).split(" ")[:3]
            else:
                lst = (" ".join(xyz[line].split())).split(" ")[:2]
        else:
            if std:
                tmplst = (" ".join(xyz[line].split())).split(" ")
                lst = [tmplst[0], tmplst[2]]
            else:
                lst = (" ".join(xyz[line].split())).split(" ")[0]
        xid.append(lst)
    f.close()
    return xid



def sae_calculator(energies, atom_count, sae_guess=np.array([-1035.30436565, -16.8356588])): #sae_guess=np.array([-1028.6, -13.8])):

    def sae_fitting(sae_values, atom_count, total_energies):
            out_eng = np.zeros(len(total_energies))
            for i in range(len(sae_values)):
                out_eng += sae_values[i]*atom_count[:,i]
            return out_eng-total_energies

    low_bounds = []
    high_bounds = []
    for i in range(len(sae_guess)):
        low_bounds.append(sae_guess[i]-8)
        high_bounds.append(sae_guess[i]+8)
    bounds=tuple([low_bounds, high_bounds])
    print('SAE bounds: ', bounds)
    print('Performing least squares fitting to get SAE values to subtact from energies.')
    lsq_data = least_squares(sae_fitting, sae_guess, bounds=bounds, args=(atom_count, energies))
    print('SAE energies: ', lsq_data.x)
    return lsq_data.x


def load_data(args, get_aevs=True, fid='', mf=None):
    print('Data random seed: ', args.randomseed[0])
    if fid=='':
        fid=args.fidlevel
    random.seed(args.randomseed[0])
    torch.manual_seed(random.randrange(200000))
    np.random.seed(random.randrange(200000))
    random.seed(random.randrange(200000))
    try:
        args.savepth = args.data_path +  '/'
    except:
        args.savepth = args.savenm +  '/'
    # Set path
    args.trpath = args.savepth + 'training'+str(fid)+'/'
    #args.vlpath = args.savepth + 'validation'+str(args.fidlevel)+'/'
    args.tspath = args.savepth + 'testing'+str(fid)+'/'
    if args.pre_saved:
        print('args.trpath: ', args.trpath)
        train_vld_engs = torch.load(args.trpath+'train_engs')

        if sum(args.tvt) > 1.0:
            trv_indx = list(range(int(args.tvt[0] + args.tvt[1])))
            random.shuffle(trv_indx)
            train_indx = trv_indx[:int(args.tvt[0])]
            valid_indx = trv_indx[int(args.tvt[0]):]
        elif sum(args.tvt)==1.0:
            ntv = len(train_vld_engs)
            trv_indx = list(range(ntv))
            random.shuffle(trv_indx)
            nt = int(args.tvt[0] * ntv)
            nv = ntv - nt
            train_indx=trv_indx[:nt]
            valid_indx=trv_indx[nv:]
        else:
            print('args.tvt must either sum to one or contain only whole numbers')
            sys.exit()

        
        del trv_indx

        if get_aevs:
            sae_energies = np.load(args.trpath + 'sae_energies.npy')
            dimdat = torch.load(args.trpath+'aev_length')
            train_vld_aevs = torch.load(args.trpath+'train_aevs')
            train_aevs = [train_vld_aevs[i] for i in train_indx]
            valid_aevs = [train_vld_aevs[i] for i in valid_indx]
            #print(train_aevs)
            del train_vld_aevs
            test_aevs = torch.load(args.tspath+'test_aevs')

        #train_vld_engs = torch.load(args.trpath+'train_engs')
        train_engs = [train_vld_engs[i] for i in train_indx]#train_vld_engs[train_indx]
        valid_engs = [train_vld_engs[i] for i in valid_indx]#train_vld_engs[valid_indx]
        del train_vld_engs
        

        test_engs = torch.load(args.tspath+'test_engs')

        if args.floss:
            train_vld_forces = torch.load(args.trpath+'train_forces')
            train_vld_fdims = torch.load(args.trpath+'train_fdims')
            train_forces = [train_vld_forces[i] for i in train_indx]
            valid_forces = [train_vld_forces[i] for i in valid_indx]

            #train_forces = train_vld_forces[train_indx]
            #valid_forces = train_vld_forces[valid_indx]
            del train_vld_forces

            train_fdims = [train_vld_fdims[i] for i in train_indx]
            valid_fdims = [train_vld_fdims[i] for i in valid_indx]
            #train_fdims = train_vld_fdims[train_indx]
            #valid_fdims = train_vld_fdims[valid_indx]
            del train_vld_fdims

            test_forces = torch.load(args.tspath+'test_forces')
            test_fdims = torch.load(args.tspath+'test_fdims')
            if get_aevs:
                #train_vld_daevs = np.array(torch.load(args.trpath+'train_daevs'))
                train_vld_daevs = torch.load(args.trpath+'train_daevs')
                train_daevs = [train_vld_daevs[i] for i in train_indx]
                valid_daevs = [train_vld_daevs[i] for i in valid_indx]


                #train_daevs = train_vld_daevs[train_indx]
                #valid_daevs = train_vld_daevs[valid_indx]
                del train_vld_daevs

                test_daevs = torch.load(args.tspath+'test_daevs')

        if get_aevs:
        
            if args.floss:
                train_stuff = MyTrainDataset(size=len(train_engs), aevs=train_aevs, forces=train_forces, daevs=train_daevs, fdims = train_fdims, engs=train_engs)
                valid_stuff = MyTrainDataset(size=len(valid_engs), aevs=valid_aevs, forces=valid_forces, daevs=valid_daevs, fdims = valid_fdims, engs=valid_engs)
                test_stuff = MyTrainDataset(size=len(test_engs), aevs=test_aevs, forces=test_forces, daevs=test_daevs, fdims = test_fdims, engs=test_engs)
            else:
                train_stuff = MyTrainDataset(size=len(train_engs), aevs=train_aevs, engs=train_engs)
                valid_stuff = MyTrainDataset(size=len(valid_engs), aevs=valid_aevs, engs=valid_engs)
                test_stuff = MyTrainDataset(size=len(test_engs), aevs=test_aevs, engs=test_engs)
        else:
            if args.floss:
                train_stuff = MyTrainDataset(size=len(train_engs), forces=train_forces, engs=train_engs)
                valid_stuff = MyTrainDataset(size=len(valid_engs), forces=valid_forces, engs=valid_engs)
                test_stuff = MyTrainDataset(size=len(test_engs), forces=test_forces, engs=test_engs)
            else:
                train_stuff = MyTrainDataset(size=len(train_engs), engs=train_engs)
                valid_stuff = MyTrainDataset(size=len(valid_engs), engs=valid_engs)
                test_stuff = MyTrainDataset(size=len(test_engs), engs=test_engs)


        train_set = train_stuff
        valid_set = valid_stuff
        test_set  = test_stuff
        keys = []
        for key in train_set.data_dict:
            keys.append(key)
        print('keys: ', keys)
        if get_aevs:
            return train_set, valid_set, test_set, sae_energies, dimdat, keys
        else:
            return train_set, valid_set, test_set, keys

def prep_data(args, mf=None, fid='', get_aevs=True):
    # Set path
    args.savepth = args.savenm +  '/'
    args.trpath = args.savepth + 'training'+str(fid)+'/'
    #args.vlpath = args.savepth + 'validation'+str(fid)+'/'
    args.tspath = args.savepth + 'testing'+str(fid)+'/'
    Path(args.savepth).mkdir(parents=True, exist_ok=True)
    Path(args.trpath).mkdir(parents=True, exist_ok=True)
    #Path(args.vlpath).mkdir(parents=True, exist_ok=True)
    Path(args.tspath).mkdir(parents=True, exist_ok=True)

    # Data preparation
    args.read_tvtmsk = False
    args.gen_byid = False
    if args.input_data_type == 'sqlite':
        if args.trtsid_name is not None:
            args.read_tvtmsk = True
            args.gen_byid = True
    elif args.trtsid_name is not None:
        args.read_tvtmsk = True

    # ==========================================================================================
    # instantiate data object for pes

    print("Instantiating PES data object")


    dpes = data.Data_pes(atom_types=args.present_elements)

    dpes.trid_fname = None
    args.excludexid = False
    # read database and parse it
    print("Get data")
    
    args.wrw = False
    if args.floss:
        # write AEV with force:
        args.wrf = True
    else:
        args.wrf = False

    t00 = timeit.default_timer()
    if args.gen_byid:
        print('prepper: calling read_xidtxt with args.trtsid_name[0]:',args.trtsid_name[0])
        xid = read_xidtxt(args.trtsid_name[0])
        print('prepper: calling get_data with xid length:',len(xid))
        dpes.get_data(args, xid=xid, fid = fid, get_aevs=get_aevs)
    else:
        print('prepper: prep_data: calling get_data with no xid spec')
        dpes.get_data(args, fid = fid, get_aevs=get_aevs)

    t01 = timeit.default_timer()
    print('Generating AEV (min): ', (t01 - t00) / 60)

    # do this for repeatability of random samples from pytorch
    print('random seed:', args.randomseed[0])
    random.seed(args.randomseed[0])
    torch.manual_seed(random.randrange(200000))
    np.random.seed(random.randrange(200000))
    random.seed(random.randrange(200000))


    print("prepper.py: prep_dat: Prepare training, validation, and testing data")
    dpes.md = [i[6] for i in dpes.meta]
    del dpes.meta


    if args.read_tvtmsk:
        tvtmsk = read_xidtxt(args.trtsid_name[0], tvtmsk=True)
        tvtset = np.array(tvtmsk)[:,1]
        tvtind = np.array(tvtmsk)[:,0]
        train_spots = np.where(tvtset=='2')[0]
        #valid_spots = np.where(tvtset=='0')[0]
        test_spots = np.where(tvtset=='-1')[0]
        train_ind = set(tvtind[train_spots])
        test_ind = set(tvtind[test_spots])
        print('prepper.py: prep_dat: tvt mask for testset was set based on the file: {}.'.format(args.trtsid_name))
    else:
        ntvts = len(dpes.md)
        nlist = list(range(len(dpes.md)))
        random.shuffle(nlist)
        if sum(args.tvt) > 1.0:
            train_ind=[dpes.md[i] for i in nlist[:args.tvt[0]+args.tvt[1]]]
            test_ind=[dpes.md[i] for i in nlist[args.tvt[2]:]]
        elif sum(args.tvt) == 1.0:
            ntv = int((args.tvt[0]+args.tvt[1]) * ntvts)
            nts = ntvts - ntv
            train_ind=[dpes.md[i] for i in nlist[:ntv]]
            test_ind=[dpes.md[i] for i in nlist[nts:]]
        else:
            print('args.tvt must either sum to one or contain only whole numbers')
            sys.exit()
    
    dpes.prep_training_data(train_ind, bpath = args.trpath, with_aev_data = get_aevs)

    #dpes.prep_validation_data(bpath = args.vlpath)

    dpes.prep_testing_data(test_ind, bpath = args.tspath, with_aev_data = get_aevs)

    sae_guess_dict = {
            'C':-1035.30436565,
            'H':-16.8356588,
            'O':-2043.89908480,
            'N':-1486.3292980,
            'F':-2715.285898636,
            'S':-10834.4396698,
            'Cl':-12522.6292598
            }

    if args.sae_fit:
        atom_count = []
        total_energies = []
        for co in range(len(dpes.train_engs)):
            acount = []
            for ty in range(len(dpes.train_aevs[co])):
                acount.append(len(dpes.train_aevs[co][ty]))
            atom_count.append(acount)
            total_energies.append(dpes.train_engs[co].item())
        atom_count = np.array(atom_count)
        sae_guess = []
        for i in args.present_elements:
            sae_guess.append(sae_guess_dict[i])
        sae_guess = np.array(sae_guess)
        sae_energies = sae_calculator(total_energies, atom_count, sae_guess=sae_guess)
        energies_to_subtract = np.zeros((len(dpes.train_engs)))
        for atype in range(len(sae_energies)):
            energies_to_subtract += sae_energies[atype] * atom_count[:,atype]
        subtracted_energies = total_energies - energies_to_subtract
        ymax = np.max(subtracted_energies)
        ymin = np.min(subtracted_energies)
        print('min energy: ', ymin)
        print('max energy: ', ymax)
        print('Mean of subtracted energies (should be close to 0): ', np.mean(subtracted_energies))
        print('Range of subtracted energies: ', max(subtracted_energies) - min(subtracted_energies))
        dpes.sae_energies = sae_energies
    else:
        dpes.sae_energies = np.zeros(dpes.num_nn)
    
    np.save(args.trpath + 'sae_energies', dpes.sae_energies)
    
    return


def cat_data(tocat, key, ind, device, num_spec):
    items = []
    ids = []
    nmols = []
    for batch_all in tocat:
        batch = [batch_all[i][ind] for i in range(len(batch_all))]
        #if key == 'engs' or key == 'forces' or key == 'daevs':
        if 'engs' in key or 'forces' in key:
            batch = [torch.tensor(i) for i in batch]
        if key == 'aevs' or key == 'daevs':
            ss = [[[] for mol in range(len(batch))] for spec in range(num_spec)]
            id_spec = [[] for spec in range(num_spec)]
            mol_count = 0
            for mol in range(len(batch)):
                ss_spec = []
                mol_count += 1
                for spec in range(len(batch[mol])):
                    ss[spec][mol] = torch.tensor(batch[mol][spec]).to(device).requires_grad_()
                    if key == 'aevs':
                        id_spec[spec] += [mol] * len(batch[mol][spec])
            ss = [torch.cat(s) for s in ss]
            bitem = ss
            if key == 'aevs':
                id_spec = [torch.tensor(spec).to(device) for spec in id_spec]
                ids.append(id_spec)
                nmols.append(mol_count)

        if 'forces' in key or 'engs' in key:
            bitem = torch.cat([item for item in batch]).to(device)
            if 'engs' in key:
                bitem = bitem.unsqueeze(0).T
        if 'fdims' in key:
            bitem = batch
        items.append(bitem)
    if key == 'aevs':
        return items, ids, nmols
    else:
        return items


class MyTrainDataset(Dataset):
    def __init__(self, size, **data):
        self.size = size
        item_count = 0
        self.data_dict = {}
        for key, value in data.items():
            self.data_dict[key] = [(value[i]) for i in range(size)]

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        ret_data = []
        for key in self.data_dict:
            ret_data.append(self.data_dict[key][index])
        return ret_data 


def my_collate(batch):
    ret_data = []
    for dp in batch:
        item = [dp[i] for i in range(len(dp))]
        ret_data.append(item)

    return ret_data



def load_train_data(args, get_aevs=True, fid=''):
    random.seed(args.randomseed[1])
    torch.manual_seed(random.randrange(200000))
    np.random.seed(random.randrange(200000))
    random.seed(random.randrange(200000))
    train_set, valid_set, test_set, keys = load_data(args, mf=None, get_aevs=get_aevs, fid=fid)
    if args.ddp:
        train_set = DataLoader(train_set, batch_size=args.tr_batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(train_set), collate_fn=my_collate)
    else:
        train_set = DataLoader(train_set, batch_size=args.tr_batch_size, pin_memory=True, shuffle=False, collate_fn=my_collate)
    valid_set = DataLoader(valid_set, batch_size=args.vl_batch_size, pin_memory=True, shuffle=False, collate_fn=my_collate)
    test_set = DataLoader(test_set, batch_size=args.ts_batch_size, pin_memory=True, shuffle=False, collate_fn=my_collate)

    return train_set, valid_set, test_set, keys

def load_train_objs(args, get_aevs=True, fid=''):
    random.seed(args.randomseed[1])
    torch.manual_seed(random.randrange(200000))
    np.random.seed(random.randrange(200000))
    random.seed(random.randrange(200000))
    #print('ABOUT TO PREP DATA!')
    #load_train_data(args)
    train_set, valid_set, test_set, sae_energies, dimdat, keys = load_data(args, get_aevs=True, fid=fid, mf=None)

    #print('DATA PREPED!')
    if args.ddp:
        train_set = DataLoader(train_set, batch_size=args.tr_batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(train_set), collate_fn=my_collate)
    else:
        train_set = DataLoader(train_set, batch_size=args.tr_batch_size, pin_memory=True, shuffle=False, collate_fn=my_collate)
    valid_set = DataLoader(valid_set, batch_size=args.vl_batch_size, pin_memory=True, shuffle=False, collate_fn=my_collate)
    test_set = DataLoader(test_set, batch_size=args.ts_batch_size, pin_memory=True, shuffle=False, collate_fn=my_collate)
    #print('train_set: ', train_set)
    print('DATA LOADED!')
    if args.load_model:
        nn_pes, optimizer, lr_scheduler = prep_model(args, sae_energies=sae_energies)
    else:
        nn_pes, optimizer, lr_scheduler      = prep_model(args,
                                                  num_nn=2, din_nn=dimdat,
                                                  my_neurons_a=args.my_neurons,
                                                  my_neurons_b=args.my_neurons,
                                                  sae_energies=sae_energies, biases=False)
    
    #del dpes
    return train_set, valid_set, test_set, nn_pes, optimizer, lr_scheduler, sae_energies, keys


# ====================================================================================================
def main():
    # set default pytorch as double precision
    #fname = 'ddp_' + str(rank) + '.log'
    random.seed(0)
    torch.manual_seed(random.randrange(200000))
    np.random.seed(random.randrange(200000))
    random.seed(random.randrange(200000))
    torch.set_default_dtype(torch.float64)
    args = parse_arguments_list()
    prep_data(args, mf=None)
    print('DONE!!!!!!!')

if __name__ == "__main__":
    tt1 = time.time()
    random.seed(0)
    torch.manual_seed(random.randrange(200000))
    np.random.seed(random.randrange(200000))
    random.seed(random.randrange(200000))
    #mp.spawn(main, args=[world_size], nprocs=world_size, join=True)
    main()
    tt2 = time.time()
    print('Time it took: ', tt2-tt1)

