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

from scipy.optimize import least_squares
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from ase.units import mol, kcal

from pykinml import data
from pykinml import nnpes
from pykinml.data import sample_xid
from pykinml import daev

batch_list_shuffle = False
verbose = False
printlr = False
write_tvtmsk = False
savetrid=False


# ===================================================================================================
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

    def get_precisions(self):
        """
        A factor of 0.5 has been factored out and ignored as it scales all terms equally.
        """
        return 1 / torch.exp(self.log_sigma)
        #return 1 / self.log_sigma**2

    def forward(self, *loss_terms):
        """
        When using traditional weighting and task weight optimization the task weights should be clamped to between 0.001 and 0.999
        to prevent negative contributions to the loss. This problem cn occur with non traditional weighting as well. Looking for way around it...
        """
        total_loss = 0
        if self.traditional:
            #total_loss += torch.clamp(1-torch.sum(self.log_sigma), 0.001, 0.999) * loss_terms[0]
            total_loss += (1-torch.sum(self.log_sigma)) * loss_terms[0]
            for task in range (1, self.num_tasks):    
                #total_loss += torch.clamp(self.log_sigma[task-1], 0.001, 0.999) * loss_terms[task]
                total_loss += self.log_sigma[task-1] * loss_terms[task]
        else:
            self.precisions = self.get_precisions()
            for task in range(self.num_tasks):
                total_loss += self.precisions[task] * loss_terms[task] + self.log_sigma[task]
                #total_loss += self.precisions[task] * loss_terms[task] + torch.log(self.log_sigma[task])
        return total_loss


# ====================================================================================================
class lossfunction():
    def __init__(self, wloss, floss, dEsq=1., dElsq=1., dfsq=1., c=1., a=0.5, w=1., p=2):
        """
        Loss function class

        wloss: weighted loss
        floss: force training
        dEsq:  (Emax - Emin)**2
        dElsq: (Emax_lf - Emin_lf)**2
        dfsq:  (Fmax - Fmin)**2
        c:     loss weight for low fidelity (cost ratio)
        a:     loss weight for force
        w:     weight vectors for weighted loss function
        p:     p-norm for loss term
        """

        self.dEsq = dEsq
        self.dElsq = dElsq
        self.dfsq = dfsq
        self.c = c
        self.a = a
        self.w = w
        if wloss:
            if floss:
                self.lossfn = self.my_loss_wf
            else:
                self.lossfn = self.my_loss_w
        elif floss:
            self.lossfn = self.my_loss_f
        else:
            self.lossfn = self.my_loss

    def my_loss(self, ph, yh, pl=None, yl=None):
        if pl != None:
            loss = torch.mean((pl - yl) ** self.p) / self.dEsq + self.c * torch.mean((ph - yh) ** self.p) / self.dElsq
        else:
            loss = torch.mean((ph - yh) ** self.p) / self.dEsq
        return loss

    def my_loss_w(self, ph, yh, pf=None, yf=None, pl=None, yl=None, pfl=None, yfl=None):
        ls = 0.
        if pl != None:
            for i in range(ph.shape[0]):
                ls = ls + self.w[i] * ((ph[i] - yh[i]) ** self.p + (pl[i] - yl[i]) ** self.p)
        else:
            for i in range(ph.shape[0]):
                ls = ls + self.w[i] * (ph[i] - yh[i]) ** self.p
        return ls[0]

    def my_loss_wf(self, ph, yh, pf, yf, pl=None, yl=None, pfl=None, yfl=None):
        ls_f = torch.tensor(0.)
        ls_e = torch.tensor(0.)
        if pl != None:
            ls_el = torch.tensor(0.)
            ls_eh = torch.tensor(0.)
            for i in range(ph.shape[0]):
                ls_el = ls_el + self.w[i] * ((pl[i] - yl[i]) ** self.p)
                ls_eh = ls_eh + self.w[i] * ((ph[i] - yh[i]) ** self.p)
                ls_f = ls_f + torch.mean((pfl[i] - yfl[i]) ** self.p) + torch.mean((pf[i] - yf[i]) ** self.p)
            ls_e = ls_el / self.dElsq + ls_eh / self.dEsq
        else:
            for i in range(ph.shape[0]):
                ls_e = ls_e + self.w[i] * ((ph[i] - yh[i]) ** self.p)
                ls_f = ls_f + self.w[i] * torch.mean((pf[i] - yf[i]) ** self.p)
            ls_e = ls_e / self.dEsq
        ls_e = ls_e[0] / ph.shape[0]
        ls_f = ls_f / ph.shape[0] / self.dfsq
        loss = (1. - self.a) * ls_e + self.a * ls_f
        return loss[0], ls_e.detach(), ls_f.detach()

    def my_loss_f(self, ph, yh, pf=None, yf=None, pl=None, yl=None, pfl=None, yfl=None):
        ls_f = torch.tensor(0.)
        if pl != None:
            ls_e = torch.mean((pl - yl) ** self.p) / self.dElsq + self.c * torch.mean((ph - yh) ** self.p) / self.dEsq
            for i in range(ph.shape[0]):
                ls_f = ls_f + torch.mean((pfl[i] - yfl[i]) ** self.p) + self.c * torch.mean((pf[i] - yf[i]) ** self.p)
        else:
            ls_e = torch.mean((ph - yh) ** self.p) / self.dEsq
            for i in range(ph.shape[0]):
                ls_f = ls_f + torch.mean((pf[i] - yf[i]) ** self.p)
        ls_f = ls_f / ph.shape[0] / self.dfsq
        loss = (1. - self.a) * ls_e + self.a * ls_f
        return loss, ls_e.detach(), ls_f.detach()

def my_loss(ph, yh, pl=None, yl=None, dEsq=1., dElsq=1., c=1.):
    if pl != None:
        loss = torch.mean((pl - yl) ** 2)/dEsq + c*torch.mean((ph - yh) ** 2)/dElsq
    else:
        loss = torch.mean((ph - yh) ** 2)/dEsq
    return loss


def my_loss_w(w, ph, yh, pl=None, yl=None):
    ls = 0.
    if pl != None:
        for i in range(ph.shape[0]):
            ls = ls + w[i] * ((ph[i] - yh[i]) ** 2 + (pl[i] - yl[i]) ** 2)
    else:
        for i in range(ph.shape[0]):
            ls = ls + w[i] * (ph[i] - yh[i]) ** 2
    return ls[0]

def my_loss_wf(a, w, ph, yh, pf, yf, pl=None, yl=None, pfl=None, yfl=None, dEsq=1., dElsq=1., dfsq=1.):
    ls_f = torch.tensor(0.)
    ls_e = torch.tensor(0.)
    if pl != None:
        ls_el = torch.tensor(0.)
        ls_eh = torch.tensor(0.)
        for i in range(ph.shape[0]):
            ls_el = ls_el + w[i] * ((pl[i] - yl[i]) ** 2)
            ls_eh = ls_eh + w[i] * ((ph[i] - yh[i]) ** 2)
            ls_f = ls_f + torch.mean((pfl[i] - yfl[i]) ** 2) + torch.mean((pf[i] - yf[i]) ** 2)
        ls_e = ls_el/dElsq + ls_eh/dEsq
    else:
        for i in range(ph.shape[0]):
            ls_e = ls_e + w[i] * ((ph[i] - yh[i]) ** 2)
            ls_f = ls_f + w[i] * torch.mean((pf[i] - yf[i]) ** 2)
        ls_e = ls_e/dEsq
    ls_e = ls_e[0]/ph.shape[0]
    ls_f = ls_f / ph.shape[0] / dfsq
    #loss = (1.-a)*ls_e + a*ls_f
    return ls_e, ls_f

def my_loss_f(a, ph, yh, pf, yf, pfdims, pl=None, yl=None, pfl=None, yfl=None, dEsq=1., dElsq=1., dfsq=1., p=2, c=1.):
    ls_f = torch.tensor(0.)
    if pl != None:
        ls_e = torch.mean((pl - yl) ** p)/dElsq + c*torch.mean((ph - yh) ** p)/dEsq
        for i in range(ph.shape[0]):
            ls_f = ls_f + torch.mean((pfl[i] - yfl[i]) ** p) + c*torch.mean((pf[i] - yf[i]) ** p)
    else:
        ls_e = torch.mean((ph - yh) ** p)/dEsq
        for i in range(ph.shape[0]):
            ls_f = ls_f + torch.mean((pf[i][0:pfdims[0][i]] - yf[i][0:pfdims[0][i]]) ** p)
            #ls_f = ls_f + torch.mean((pf[i] - yf[i]) ** p)
    ls_f = ls_f/ph.shape[0]/dfsq
    #loss = (1.-a)*ls_e + a*ls_f
    return ls_e, ls_f



def my_loss_f_rel(a, ph, yh, pf, yf, pfdims, expw, pl=None, yl=None, pfl=None, yfl=None, dEsq=1., dElsq=1., dfsq=1., p=2, c=1.):
    ls_f = torch.tensor(0.)
    if pl != None:
        ls_e = torch.mean((pl - yl) ** p)/dElsq + c*torch.mean((ph - yh) ** p)/dEsq
        for i in range(ph.shape[0]):
            ls_f = ls_f + torch.mean((pfl[i] - yfl[i]) ** p) + c*torch.mean((pf[i] - yf[i]) ** p)
    else:
        ls_e = torch.mean(((ph - yh) ** p) * expw)/dEsq
        for i in range(ph.shape[0]):
            ls_f = ls_f + torch.mean(((pf[i][0:pfdims[0][i]] - yf[i][0:pfdims[0][i]]) ** p) * expw[i])
            #ls_f = ls_f + torch.mean((pf[i] - yf[i]) ** p)
    ls_f = ls_f/ph.shape[0]/dfsq
    #loss = (1.-a)*ls_e + a*ls_f
    return ls_e, ls_f




def changelr(optimizer, newlr):
    for g in optimizer.param_groups:
        g['lr'] = newlr
    return 0

# ====================================================================================================
def prep_model(load_model, args, load_opt=True,
               num_nn=None, din_nn=256,
               my_neurons_a=None, my_activations_a=None,
               my_neurons_b=None, my_activations_b=None,
               my_neurons_hf_a=None, my_activations_hf_a=None,
               my_neurons_hf_b=None, my_activations_hf_b=None,
               sae_energies=np.zeros(2), biases=True):
    #print("Defining neural networks ")
    if not load_model and my_neurons_a == None:
        print('Specify the shape of input, number of hidden neurons and activation functions.')
        sys.exit()
    #if not load_opt:
        #print('optimizer is not loaded.')

    # arch could be 'hfonly', 'seq-2net', 'seq-1net', 'hybrid-seq-2net', 'hybrid-dscr', or 'dscr'.
    if not load_model:
        arch = args.fid[0]
        args.netparams = {'arch': arch,
                          'n_input_a': din_nn, 'n_input_b': din_nn,
                          'neurons_a': my_neurons_a, 'activations_a': my_activations_a,
                          'neurons_b': my_neurons_b, 'activations_b': my_activations_b,
                          'neurons_hf': my_neurons_hf_a, 'activations_hf': my_activations_hf_a,
                          'neurons_hf_b': my_neurons_hf_b, 'activations_hf_b': my_activations_hf_b,
                          'sae_energies': sae_energies, 'add_sae': False, 'biases': biases
                          }
        net = nnpes.CompositeNetworks(**args.netparams)
        net.sae_energies = sae_energies
        print('net:', net)
        args.netparams['activations_a'] = args.my_actfn
        args.netparams['activations_b'] = args.my_actfn
        args.netparams['activations_hf'] = args.my_actfn_hf
        args.netparams['activations_hf_b'] = args.my_actfn_hf
        #print('net.state_dict:', net.state_dict())
        #==========================================================================================
        # experiment with weight/bias initializations
        # https://discuss.pytorch.org/t/linear-layer-default-weight-initialization/23610
        # This code demonstrates sampling NN weights/biases from alternate distributions
        # It works only for the single fidelity case
        # The default in pytorch is that weights/biases are sampled from U(-a,a) where
        # a = 1/sqrt(fan_in), where fan_in is the input-dimensionality for the layer
        # a = 10 fixes convergence issues

        if args.netparam_init_amp != 1.0:
            print("args.netparam_init_amp:",args.netparam_init_amp)
            if args.netparam_init_amp <= 0:
                print("Error: args.netparam_init_amp must me greater than 0")
                sys.exit(1)
            uamp = args.netparam_init_amp  # default 1.0
            for nnt in net.children():
                for linblock in nnt:
                    ablk  = uamp/math.sqrt(linblock.linear.in_features)
                    nn.init.uniform_(linblock.linear.weight, -ablk, ablk)
                    nn.init.uniform_(linblock.linear.bias, -ablk, ablk)
                    print('linblock linear (',linblock.linear.in_features,'x',linblock.linear.out_features,') =',linblock.linear.in_features*linblock.linear.out_features,' weights:',linblock.linear.weight)
                    print('linblock linear',linblock.linear.out_features,'biases:',linblock.linear.bias)

        #==========================================================================================

        nn_pes = Model(net)
        if args.optimizer[0] == 'SGD':
            optimizer = optim.SGD(nn_pes.net.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                  nesterov=True, weight_decay=args.weight_decay)
        elif args.optimizer[0] == 'Adam':
            optimizer = optim.Adam(nn_pes.net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer[0] == 'AdamW':
            optimizer = optim.AdamW(nn_pes.net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        #================================================================================================

    else:
        if '.pt' not in args.load_model_name[0]:
            fns_par = [os.path.basename(x) for x in glob(args.load_model_name[0]+'*.pt')]
            iterlist = [int(x.split('-')[-1].split('.')[0]) for x in fns_par]
            try:
                max_iter = max(iterlist)
            except:
                print('No model file was found in {}. It is empty.'.format(args.load_model_name[0]))
                sys.exit()
            if max_iter < 1000:
                max_iter = str(max_iter).zfill(4)
            loadnm = args.load_model_name[0] + 'comp-' + str(max_iter) + '.pt'
        else:
            loadnm = args.load_model_name[0]
        checkpoint = torch.load(loadnm, map_location=torch.device(args.device))
        checkpoint['params']['activations_a']  = args_to_actfn(checkpoint['params']['activations_a'])
        checkpoint['params']['activations_b']  = args_to_actfn(checkpoint['params']['activations_b'])

        
        try:
            log_sigma = checkpoint['model_state_dict'].pop('log_sigma', None)
        except:
            pass

        try:
            args.netparams = checkpoint['params']
            args.netparams['add_sae']=True
            net = nnpes.CompositeNetworks(**args.netparams)
            #net['sae_energies'] = sae_energies 

        except:
            print('NN architecture was not saved in the comp.pt. NN was generated using preset parameters.')
            print(
                'preset: neurons = [128, 128, 64, 1], activation = [gaussian gaussian gaussian identity].')
            print('preset: optimizer = SGD(lr=0.001, momentum=0.5, nesterov=True)')
            my_neurons = args.my_neurons
            my_activations = [nnpes.gaussian, nnpes.gaussian, nnpes.gaussian, nnpes.identity]
            args.netparams = {'arch': args.fid[0],
                              'n_input_a': din_nn, 'n_input_b': din_nn,
                              'neurons_a': my_neurons, 'activations_a': my_activations,
                              'neurons_b': my_neurons, 'activations_b': my_activations,
                              'neurons_hf': my_neurons, 'activations_hf': my_activations,
                              'neurons_hf_b': my_neurons, 'activations_hf_b': my_activations,
                              'sae_energies': sae_energies, 'add_sae': True
                              }
            net = nnpes.CompositeNetworks(**args.netparams)

        nn_pes = Model(net)
        nn_pes.net.load_state_dict(checkpoint['model_state_dict'])

        if load_opt:
            lr_loaded = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
            if args.learning_rate == 0:
                try:
                    lr_set = lr_loaded
                except:
                    lr_set = args.learning_rate
            else:
                if lr_loaded != args.learning_rate:
                    print(
                        'The loaded learning rate is {} but the parsed learning rate is {}'.format(lr_loaded,
                                                                                                   args.learning_rate))
                lr_set = args.learning_rate

            try:
                args.crt_err = checkpoint['crt_err']
            except:
                args.crt_err = 1.
                print('crt err was not saved. default is 1.')

            try:
                if args.weight_decay != checkpoint['weight_decay']:
                    print('the loaded L2-penalty is {} but the set value is {}.'.format(checkpoint['weight_decay'],
                                                                                        args.weight_decay))
                else:
                    print('loaded L2 penalty: ', args.weight_decay)
            except:
                print('L2-penalty was not saved. The set value is {}.'.format(args.weight_decay))

            if nn_pes.net.arch == 'hfonly':
                optpars = nn_pes.net.parameters()
                # optpars = [
                #     {'params': nn_pes.net.NNa.parameters(), 'lr': lr_set},
                #     {'params': nn_pes.net.NNb.parameters(), 'lr': lr_set}
                # ]
            else:
                if args.refinehf:
                    if nn_pes.net.arch == 'seq-1net':
                        optpars = [{'params': nn_pes.net.NN_hf.parameters(), 'lr': lr_set, 'weight_decay': args.weight_decay}]
                    else:
                        optpars = [
                            {'params': nn_pes.net.NN_hf.parameters(), 'lr': lr_set, 'weight_decay': args.weight_decay},
                            {'params': nn_pes.net.NN_hf_b.parameters(), 'lr': lr_set, 'weight_decay': args.weight_decay}
                        ]
                elif args.refinelf:
                    optpars = [
                        {'params': nn_pes.net.NNa.parameters(), 'lr': lr_set, 'weight_decay': args.weight_decay},
                        {'params': nn_pes.net.NNb.parameters(), 'lr': lr_set, 'weight_decay': args.weight_decay}
                    ]
                else:
                    if nn_pes.net.arch == 'seq-1net':
                        optpars = [
                            {'params': nn_pes.net.NNa.parameters(), 'lr': lr_set, 'weight_decay': args.weight_decay},
                            {'params': nn_pes.net.NNb.parameters(), 'lr': lr_set, 'weight_decay': args.weight_decay},
                            {'params': nn_pes.net.NN_hf.parameters(), 'lr': lr_set, 'weight_decay': args.weight_decay}
                        ]
                    else:
                        optpars = [
                            {'params': nn_pes.net.NNa.parameters(), 'lr': lr_set, 'weight_decay': args.weight_decay},
                            {'params': nn_pes.net.NNb.parameters(), 'lr': lr_set, 'weight_decay': args.weight_decay},
                            {'params': nn_pes.net.NN_hf.parameters(), 'lr': lr_set, 'weight_decay': args.weight_decay},
                            {'params': nn_pes.net.NN_hf_b.parameters(), 'lr': lr_set, 'weight_decay': args.weight_decay}
                        ]

            
            # optpars = nn_pes.net.parameters()
            if args.optimizer[0] == 'SGD':
                optimizer = optim.SGD(optpars, momentum=args.momentum, nesterov=True, lr=lr_set, weight_decay=args.weight_decay)
            elif args.optimizer[0] == 'Adam':
                optimizer = optim.Adam(optpars, lr=lr_set, weight_decay=args.weight_decay)
            elif args.optimizer[0] == 'AdamW':
                optimizer = optim.AdamW(optpars, lr=lr_set, weight_decay=args.weight_decay)

            
            #================================================================================================
            
            num_nn = len([chi for chi in net.named_children()])
            num_pars = int(checkpoint['optimizer_state_dict']['param_groups'][0]['params'].__len__() / num_nn)

            try:
                opt_state_dict_group = checkpoint['optimizer_state_dict']
                opttype = checkpoint['optimizer'].__module__.split('.')[-1]
                if str.lower(args.optimizer[0]) == opttype:
                    state = optimizer.state_dict()
                    state.update(opt_state_dict_group)
                    optimizer.load_state_dict(state)
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(args.device)
                    print('{} parameters were loaded.'.format(loadnm))

                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_set
                        param_group['weight_decay'] = args.weight_decay
                else:
                    print('optimizer in {} is {} but {} was set. {} will be used for training.'.format(
                        args.load_model_name[0], opttype, args.optimizer[0], args.optimizer[0]))
            except:
                print('# of hyperparameters were not matched for the current optimizer and saved optimizer ({}). Optimizer setting was reset.'.format(loadnm))

            if args.optimize_force_weight:
                nn_pes.net.log_sigma = log_sigma
                optimizer.param_groups[0]['params'].append(log_sigma)


    # define criterion
    try:
        args.crt_err = checkpoint['crt_err']
    except:
        args.crt_err = 1.
        print('crt err was not saved. default is 1.')
    
    criterion = my_loss
    if load_opt:
        return nn_pes, criterion, optimizer
    else:
        return nn_pes

# ====================================================================================================

def print_opt(optimizer, opt_fnam, epoch):
    fname = opt_fnam + "-" + str(epoch).zfill(4) + ".dat"
    with open(fname, 'w') as f:
        torch.set_printoptions(profile="full", precision=14, sci_mode=True)
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name], file=f)
        torch.set_printoptions(profile="default")



# ====================================================================================================
class Model():
    def __init__(self, net):
        """
        NN PES model class
        """
        self.net = net
        self.tg = np.zeros(2)

    def print_model(self, net_fnam, epoch):
        fname = net_fnam + "-" + str(epoch).zfill(4) + ".dat"
        with open(fname, 'w') as f:
            torch.set_printoptions(profile="full", precision=14, sci_mode=True)
            for param_tensor in self.net.state_dict():
                print(param_tensor, "\t", self.net.state_dict()[param_tensor], file=f)
            torch.set_printoptions(profile="default")
        if epoch == 0:
            for param_tensor in self.net.state_dict():
                fname = net_fnam + "-" + str(epoch).zfill(4) + "-" + param_tensor + ".dat"
                with open(fname, 'w') as f:
                    torch.set_printoptions(profile="full", precision=14, sci_mode=True)
                    np.savetxt(fname, self.net.state_dict()[param_tensor].numpy())
                    torch.set_printoptions(profile="default")

    def save(self, args, net_fnam, epoch, optimizer, crt_err=1.):
        torch.save({'epoch': epoch,
                    #'net': self.net,
                    'optimizer': optimizer,
                    'params': args.netparams,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'crt_err': crt_err,
                    'weight_decay': args.weight_decay
                    },
                   net_fnam + '-' + str(epoch).zfill(4) + ".pt")

    def eval_dl(self, dpes, b=0):
        """
        xb is the AEV data for a batch of training configurations/data-points
        Each element x[t] is a 2d tensor for atom-type t
                it has some Nb rows ... being the number of type-t atoms-AEVs in this batch
                it has some laev columns ... being the size of the AEV
        thus loop over x with something like: for xt in x:
            x[0], x[1], etc are the data tensors for each nn 0, 1,
            ... corresp to each atom type, for this batch
        net is a list of NN objects
        ii is an index array for this batch, so for t=0,1 ii could be something like this, for a
            context where this batch corresponds to 2 data points
            ii[0]: [ [0, 1] , [2, 3, 4] ]
            ii[1]: [ [0]    , [1,2]]
          In this context with 2 data points, we have two items in each ii[t]
          Each of these is a list of monotonically increasing integers,
            being the indices for rows in x[t] that correspond to each of the two data points
          Thus, in the above example, for t=0, x[0] is a tensor with 5 rows, and laev columns.
            The first two columns pertain to data point 1, and the subsequent three rows
            pertain to data point 2. Thus, data point 1 involves two atoms, while data point 2 has 3 atoms.
          Accordingly, the utility of ii is to encode the mapping from N data points, where each can involve
            a different number of atoms of each type, to a rectangular tensor with a given number of rows,
            each of which is laev long, and back. The inverse map is crucial, as the net outputs for each type t
            need to be added together within each list in ii[t]. Thus for example, above, for t=0, the net(0)
            output will have 5 rows, and 1 column. The first two rows (indexed by [0,1]) need to be added together
            and the subsequent 3 rows ([2,3,4]) need to be added together, to provide two final output values for
            type 0, corresponding to each of the two data points. Similarly for type t=1.
            Finally, the t=0,1 output values are to be added together, giving a final output tensor for the two data
            points (accounting for all atom types) that is 2-rows x 1 column.
        """

        # loop over NNs and xt:=x[t] pairs
        # for each, if xt is not empty, then compute the NN output
        #           if xt is empty, then fill-in an empty tensor in the corresponding element of the output
        # the result, "out" is a torch tensor that has the forward NN prediction for xt
        #      each row of out is the output for the corresponding row in xt
        #      out has 1 column only

        if verbose:
            print("xb[b]:")
            for t, xt in enumerate(dpes.xb[b]):
                print("xb[b][", t, "]:\n", xt)

        if self.net.arch == 'hfonly':
            if dpes.indvout:
                out_lf, out, out_C, out_H = self.net(Xa=dpes.xb[b][0], Xb=dpes.xb[b][1], idx=dpes.inddl[b],
                                                     indvout=dpes.indvout)
            else:
                out_lf, out, = self.net(Xa=dpes.xb[b][0], Xb=dpes.xb[b][1], idx=dpes.inddl[b])

        else:
            if dpes.indvout:
                try:
                    out_lf, out, out_C, out_H = self.net(Xa=dpes.xb_lf[b][0], Xb=dpes.xb_lf[b][1], idx=dpes.inddl_lf[b],
                                                         Xhfa=dpes.xb[b][0], Xhfb=dpes.xb[b][1], idx_hf=dpes.inddl[b],
                                                         indvout=dpes.indvout, maxmin=[dpes.ymax_lf, dpes.ymin_lf])
                except:
                    out_lf, out, out_C, out_H = self.net(Xa=dpes.xb[b][0], Xb=dpes.xb[b][1], idx=dpes.inddl[b],
                                                         Xhfa=dpes.xb[b][0], Xhfb=dpes.xb[b][1], idx_hf=dpes.inddl[b],
                                                         indvout=dpes.indvout, maxmin=[dpes.ymax, dpes.ymin])
            else:
                try:
                    out_lf, out = self.net(Xa=dpes.xb_lf[b][0], Xb=dpes.xb_lf[b][1], idx=dpes.inddl_lf[b],
                                           Xhfa=dpes.xb[b][0], Xhfb=dpes.xb[b][1], idx_hf=dpes.inddl[b], maxmin=[dpes.ymax_lf, dpes.ymin_lf])
                except:
                    out_lf, out = self.net(Xa=dpes.xb[b][0], Xb=dpes.xb[b][1], idx=dpes.inddl[b],
                                           Xhfa=dpes.xb[b][0], Xhfb=dpes.xb[b][1], idx_hf=dpes.inddl[b], maxmin=[dpes.ymax, dpes.ymin])

        if verbose:
            print("out:\n", out)

        ylf, yhf = out_lf, out

        if verbose:
            print("y:\n", yhf)

        if dpes.indvout:
            return ylf, yhf, out_C, out_H
        else:
            return ylf, yhf

    def eval_tr(self, dpes, b=0):

        t0 = timeit.default_timer()


        if self.net.arch == 'hfonly':
            out_lf, out = self.net(Xa=dpes.xbtr[b][0], Xb=dpes.xbtr[b][1], idx=dpes.indtr[b])
        else:
            out_lf, out = self.net(Xa=dpes.xbtr_lf[b][0], Xb=dpes.xbtr_lf[b][1], idx=dpes.indtr_lf[b],
                                   Xhfa=dpes.xbtr[b][0], Xhfb=dpes.xbtr[b][1], idx_hf=dpes.indtr[b], maxmin=[dpes.ymax_lf, dpes.ymin_lf])

        t1 = timeit.default_timer()

        t2 = timeit.default_timer()

        ylf, yhf = out_lf, out

        t3 = timeit.default_timer()

        self.tg[0] += t1 - t0
        self.tg[1] += t3 - t2

        return ylf, yhf

    def eval_vl(self, dpes, b=0):



        
        if self.net.arch == 'hfonly':
            out_lf, out = self.net(Xa=dpes.xbvl[b][0], Xb=dpes.xbvl[b][1], idx=dpes.indvl[b])
        else:
            out_lf, out = self.net(Xa=dpes.xbvl_lf[b][0], Xb=dpes.xbvl_lf[b][1], idx=dpes.indvl_lf[b],
                                   Xhfa=dpes.xbvl[b][0], Xhfb=dpes.xbvl[b][1], idx_hf=dpes.indvl[b], maxmin=[dpes.ymax_lf, dpes.ymin_lf])

        if verbose:
            print("out:\n", out)


        ylf, yhf = out_lf, out


        return ylf, yhf

    def eval_ts(self, dpes, b=0):


        if dpes.ntsdat == 0:
            print("Error: eval_ts: there is no testing data defined")
            # sys.exit()



        if self.net.arch == 'hfonly':
            out_lf, out = self.net(Xa=dpes.xbts[b][0], Xb=dpes.xbts[b][1], idx=dpes.indts[b])
        else:
            try:
                out_lf, out = self.net(Xa=dpes.xbts_lf[b][0], Xb=dpes.xbts_lf[b][1], idx=dpes.indts_lf[b],
                                       Xhfa=dpes.xbts[b][0], Xhfb=dpes.xbts[b][1], idx_hf=dpes.indts[b], maxmin=[dpes.ymax_lf, dpes.ymin_lf])
            except:
                out_lf, out = self.net(Xa=dpes.xbts[b][0], Xb=dpes.xbts[b][1], idx=dpes.indts[b],
                                       Xhfa=dpes.xbts[b][0], Xhfb=dpes.xbts[b][1], idx_hf=dpes.indts[b], maxmin=[dpes.ymax_lf, dpes.ymin_lf])

        if verbose:
            print("out:\n", out)


        ylf, yhf = out_lf, out

        if verbose:
            print("y:\n", yhf)

        return ylf, yhf


# ====================================================================================================
def eval_err_tr(nn_pes, dpes, bpath=None, floss=False):
    fntr = torch.tensor(float(dpes.ntrdat), device=dpes.device)
    fotens = [[] for b in range(dpes.nbttr)]
    fdtens = [[] for b in range(dpes.nbttr)]

    otens = torch.empty(dpes.ntrdat, device=dpes.device)
    dtens = torch.empty(dpes.ntrdat, device=dpes.device)

    if nn_pes.net.arch != 'hfonly':
        fntr_lf = torch.tensor(float(dpes.ntrdat_lf), device=dpes.device)
        fotens_lf = [[] for b in range(dpes.nbttr_lf)]
        fdtens_lf = [[] for b in range(dpes.nbttr_lf)]

        otens_lf = torch.empty(dpes.ntrdat_lf, device=dpes.device)
        dtens_lf = torch.empty(dpes.ntrdat_lf, device=dpes.device)
    for b in range(dpes.nbttr):                                     #loop over batches?
        out_lf, out = nn_pes.eval_tr(dpes, b)
        dtens[dpes.bitr[b]:dpes.bftr[b]] = dpes.ybtr[b][:, 0]
        otens[dpes.bitr[b]:dpes.bftr[b]] = out[:, 0]
        if floss:
            predf, truef = daev.cal_dEdxyz_tr(dpes, -out, b, bpath = bpath)
            fotens[dpes.bitr[b]:dpes.bftr[b]] = predf
            fdtens[dpes.bitr[b]:dpes.bftr[b]] = truef#dpes.fbtr[b]
        if nn_pes.net.arch != 'hfonly':
            otens_lf[dpes.bitr_lf[b]:dpes.bftr_lf[b]] = out_lf[:, 0]
            dtens_lf[dpes.bitr_lf[b]:dpes.bftr_lf[b]] = dpes.ybtr_lf[b][:, 0]
            if floss:
                predf_lf, truef_lf = daev.cal_dEdxyz_tr_lf(dpes, -out_lf, b)
                fotens_lf[dpes.bitr_lf[b]:dpes.bftr_lf[b]] = predf_lf
                fdtens_lf[dpes.bitr_lf[b]:dpes.bftr_lf[b]] = truef#dpes.fbtr_lf[b]

    if floss:
        #fot = torch.stack(fotens)
        #fdt = torch.stack(fdtens)

        fot = list(itertools.chain.from_iterable(fotens))
        fdt = list(itertools.chain.from_iterable(fdtens))
        fot2 = torch.stack(fot)
        fdt2 = torch.stack(fdt)

        #ffnts = fot.shape[0] * fot.shape[1]
        #fot2 = fot.reshape(1, ffnts)
        #fdt2 = fdt.reshape(1, ffnts)

        f_difs = fdt2-fot2
        f_avg_abs_e = torch.mean(abs(f_difs))
        f_rmse = torch.sqrt(torch.mean(f_difs**2))
        f_max_abs_e = torch.max(abs(f_difs))

        #torch.dist(otens, dtens, 2.) / math.sqrt(fntr)
        #f_rmse = torch.dist(fot2, fdt2, 2.) / torch.norm(fdt2)
        #f_avg_abs_e = torch.dist(fot2, fdt2, 1.) / torch.norm(fdt2, p=1.)
        #f_max_abs_e = torch.dist(fot2, fdt2, float("inf")) / torch.norm(fdt2, p=float("inf"))
        if nn_pes.net.arch != 'hfonly':

            fot_lf = list(itertools.chain.from_iterable(fotens_lf))
            fdt_lf = list(itertools.chain.from_iterable(fdtens_lf))
            fot2_lf = torch.stack(fot_lf)
            fdt2_lf= torch.stack(fdt_lf)

            f_difs_lf = fdt2_lf-fot2_lf
            f_avg_abs_e_lf = torch.mean(abs(f_difs_lf))
            f_rmse_lf = torch.sqrt(torch.mean(f_difs_lf**2))
            f_max_abs_e_lf = torch.max(abs(f_difs_lf))

            #fot_lf = torch.stack(fotens_lf)
            #fdt_lf = torch.stack(fdtens_lf)
            #ffntr_lf = fot_lf.shape[0] * fot_lf.shape[1]
            #fot2_lf = fot_lf.reshape(1, ffntr_lf)
            #fdt2_lf = fdt_lf.reshape(1, ffntr_lf)

            #f_rmse_lf = torch.dist(fot2_lf, fdt2_lf, 2.) / torch.norm(fdt2_lf)
            #f_avg_abs_e_lf = torch.dist(fot2_lf, fdt2_lf, 1.) / torch.norm(fdt2_lf, p=1.)
            #f_max_abs_e_lf = torch.dist(fot2_lf, fdt2_lf, float("inf")) / torch.norm(fdt2_lf, p=float("inf"))

    rmse = torch.dist(otens, dtens, 2.) / math.sqrt(fntr)
    avg_abs_e = torch.dist(otens, dtens, 1.) / fntr
    max_abs_e = torch.dist(otens, dtens, float("inf"))

    if nn_pes.net.arch != 'hfonly':
        rmse_lf = torch.dist(otens_lf, dtens_lf, 2.) / math.sqrt(fntr_lf)
        avg_abs_e_lf = torch.dist(otens_lf, dtens_lf, 1.) / fntr_lf
        max_abs_e_lf = torch.dist(otens_lf, dtens_lf, float("inf"))

    if nn_pes.net.arch != 'hfonly':
        if floss:
            return [kcpm(rmse.detach()), kcpm(avg_abs_e.detach()), kcpm(max_abs_e.detach()), kcpm(rmse_lf.detach()), kcpm(avg_abs_e_lf.detach()), kcpm(max_abs_e_lf.detach()), kcpm(f_rmse.detach()), kcpm(
                f_avg_abs_e.detach()), kcpm(f_max_abs_e.detach()), kcpm(f_rmse_lf.detach()), kcpm(f_avg_abs_e_lf.detach()), kcpm(f_max_abs_e_lf.detach())]
        else:
            return [kcpm(rmse.detach()), kcpm(avg_abs_e.detach()), kcpm(max_abs_e.detach()), kcpm(rmse_lf.detach()), kcpm(avg_abs_e_lf.detach()), kcpm(max_abs_e_lf.detach()), *[torch.tensor(0.) for i in range(6)]]
    else:
        if floss:
            return [kcpm(rmse.detach()), kcpm(avg_abs_e.detach()), kcpm(max_abs_e.detach()), *[torch.tensor(0.) for i in range(3)], kcpm(f_rmse.detach()), kcpm(
                f_avg_abs_e.detach()), kcpm(f_max_abs_e.detach()), *[torch.tensor(0.) for i in range(3)]]
        else:
            return [kcpm(rmse.detach()), kcpm(avg_abs_e.detach()), kcpm(max_abs_e.detach()), *[torch.tensor(0.) for i in range(9)]]



# ====================================================================================================
def eval_err_ts(nn_pes, dpes, bpath = None,floss=False, outlf=False):
    fnts = torch.tensor(float(dpes.ntsdat), device=dpes.device)
    fotens = [[] for b in range(dpes.nbtts)]
    fdtens = [[] for b in range(dpes.nbtts)]


    otens = torch.empty(dpes.ntsdat, device=dpes.device)
    dtens = torch.empty(dpes.ntsdat, device=dpes.device)

    if nn_pes.net.arch != 'hfonly':
        if outlf:
            fnts_lf = torch.tensor(float(dpes.ntsdat_lf), device=dpes.device)
            fotens_lf = [[] for b in range(dpes.nbtts_lf)]
            fdtens_lf = [[] for b in range(dpes.nbtts_lf)]

            otens_lf = torch.empty(dpes.ntsdat_lf, device=dpes.device)
            dtens_lf = torch.empty(dpes.ntsdat_lf, device=dpes.device)
    for b in range(dpes.nbtts):
        out_lf, out = nn_pes.eval_ts(dpes, b)
        dtens[dpes.bits[b]:dpes.bfts[b]] = dpes.ybts[b][:, 0]
        otens[dpes.bits[b]:dpes.bfts[b]] = out[:, 0]
        if floss:
            predf, truef = daev.cal_dEdxyz_ts(dpes, -out, b, bpath = bpath)
            fotens[dpes.bits[b]:dpes.bfts[b]] = predf
            fdtens[dpes.bits[b]:dpes.bfts[b]] = truef #dpes.fbts[b]
        if nn_pes.net.arch != 'hfonly':
            if outlf:
                otens_lf[dpes.bits_lf[b]:dpes.bfts_lf[b]] = out_lf[:, 0]
                dtens_lf[dpes.bits_lf[b]:dpes.bfts_lf[b]] = dpes.ybts_lf[b][:, 0]
                if floss:
                    predf_lf, truef_lf = daev.cal_dEdxyz_ts_lf(dpes, -out_lf, b)
                    fotens_lf[dpes.bits_lf[b]:dpes.bfts_lf[b]] = predf_lf
                    fdtens_lf[dpes.bits_lf[b]:dpes.bfts_lf[b]] = truef_lf#dpes.fbts_lf[b]

    if floss:
        #print('fotens: ', fotens)
        #fot = torch.stack(fotens)
        #fdt = torch.stack(fdtens)
        fot = list(itertools.chain.from_iterable(fotens))
        fdt = list(itertools.chain.from_iterable(fdtens))
        fot2 = torch.stack(fot)
        fdt2 = torch.stack(fdt)
        
        #ffnts = fot.shape[0] * fot.shape[1]
        #fot2 = fot.reshape(1, ffnts)
        #fdt2 = fdt.reshape(1, ffnts)
        
        f_difs = fdt2-fot2
        f_avg_abs_e = torch.mean(abs(f_difs))
        f_rmse = torch.sqrt(torch.mean(f_difs**2))
        f_max_abs_e = torch.max(abs(f_difs))
        
        #f_rmse = torch.dist(fot2, fdt2, 2.) / torch.norm(fdt2)
        #f_avg_abs_e = torch.dist(fot2, fdt2, 1.) / torch.norm(fdt2, p=1.)
        #f_max_abs_e = torch.dist(fot2, fdt2, float("inf")) / torch.norm(fdt2, p=float("inf"))

        if nn_pes.net.arch != 'hfonly':
            if outlf:

                fot_lf = list(itertools.chain.from_iterable(fotens_lf))
                fdt_lf = list(itertools.chain.from_iterable(fdtens_lf))
                fot2_lf = torch.stack(fot_lf)
                fdt2_lf = torch.stack(fdt_lf)


                f_difs_lf = fdt2_lf-fot2_lf
                f_avg_abs_e_lf = torch.mean(abs(f_difs_lf))
                f_rmse_lf = torch.sqrt(torch.mean(f_difs_lf**2))
                f_max_abs_e_lf = torch.max(abs(f_difs_lf))

                #fot_lf = torch.stack(fotens_lf)
                #fdt_lf = torch.stack(fdtens_lf)
                #ffnts_lf = fot_lf.shape[0] * fot_lf.shape[1]
                #fot2_lf = fot_lf.reshape(1, ffnts_lf)
                #fdt2_lf = fdt_lf.reshape(1, ffnts_lf)

                #f_rmse_lf = torch.dist(fot2_lf, fdt2_lf, 2.) / torch.norm(fdt2_lf)
                #f_avg_abs_e_lf = torch.dist(fot2_lf, fdt2_lf, 1.) / torch.norm(fdt2_lf, p=1.)
                #f_max_abs_e_lf = torch.dist(fot2_lf, fdt2_lf, float("inf")) / torch.norm(fdt2_lf, p=float("inf"))

    rmse = torch.dist(otens, dtens, 2.) / math.sqrt(fnts)
    avg_abs_e = torch.dist(otens, dtens, 1.) / fnts
    max_abs_e = torch.dist(otens, dtens, float("inf"))
    if nn_pes.net.arch != 'hfonly':
        if outlf:
            rmse_lf = torch.dist(otens_lf, dtens_lf, 2.) / math.sqrt(fnts_lf)
            avg_abs_e_lf = torch.dist(otens_lf, dtens_lf, 1.) / fnts_lf
            max_abs_e_lf = torch.dist(otens_lf, dtens_lf, float("inf"))

    if nn_pes.net.arch != 'hfonly':
        if outlf:
            if floss:
                return [kcpm(rmse.detach()), kcpm(avg_abs_e.detach()), kcpm(max_abs_e.detach()), kcpm(rmse_lf.detach()),
                        kcpm(avg_abs_e_lf.detach()), kcpm(max_abs_e_lf.detach()), kcpm(f_rmse.detach()), kcpm(
                        f_avg_abs_e.detach()), kcpm(f_max_abs_e.detach()), kcpm(f_rmse_lf.detach()),
                        kcpm(f_avg_abs_e_lf.detach()), kcpm(f_max_abs_e_lf.detach())]
            else:
                return [kcpm(rmse.detach()), kcpm(avg_abs_e.detach()), kcpm(max_abs_e.detach()), kcpm(rmse_lf.detach()),
                        kcpm(avg_abs_e_lf.detach()), kcpm(max_abs_e_lf.detach()), *[torch.tensor(0.) for i in range(6)]]
        else:
            if floss:
                return [kcpm(rmse.detach()), kcpm(avg_abs_e.detach()), kcpm(max_abs_e.detach()),
                        *[torch.tensor(0.) for i in range(3)], kcpm(f_rmse.detach()), kcpm(
                        f_avg_abs_e.detach()), kcpm(f_max_abs_e.detach()), *[torch.tensor(0.) for i in range(3)]]
            else:
                return [kcpm(rmse.detach()), kcpm(avg_abs_e.detach()), kcpm(max_abs_e.detach()),
                        *[torch.tensor(0.) for i in range(9)]]
    else:
        if floss:
            return [kcpm(rmse.detach()), kcpm(avg_abs_e.detach()), kcpm(max_abs_e.detach()),
                    *[torch.tensor(0.) for i in range(3)], kcpm(f_rmse.detach()), kcpm(
                    f_avg_abs_e.detach()), kcpm(f_max_abs_e.detach()), *[torch.tensor(0.) for i in range(3)]]
        else:
            return [kcpm(rmse.detach()), kcpm(avg_abs_e.detach()), kcpm(max_abs_e.detach()),
                    *[torch.tensor(0.) for i in range(9)]]




# ====================================================================================================
def eval_err_vl(nn_pes, dpes, bpath=None, floss=False):
    fnvl = torch.tensor(float(dpes.nvldat), device=dpes.device)
    fotens = [[] for b in range(dpes.nbtvl)]
    fdtens = [[] for b in range(dpes.nbtvl)]

    otens = torch.empty(dpes.nvldat, device=dpes.device)
    dtens = torch.empty(dpes.nvldat, device=dpes.device)

    if nn_pes.net.arch != 'hfonly':
        fnvl_lf = torch.tensor(float(dpes.nvldat_lf), device=dpes.device)
        fotens_lf = [[] for b in range(dpes.nbtvl_lf)]
        fdtens_lf = [[] for b in range(dpes.nbtvl_lf)]

        otens_lf = torch.empty(dpes.nvldat_lf, device=dpes.device)
        dtens_lf = torch.empty(dpes.nvldat_lf, device=dpes.device)

    for b in range(dpes.nbtvl):
        out_lf, out = nn_pes.eval_vl(dpes, b)
        dtens[dpes.bivl[b]:dpes.bfvl[b]] = dpes.ybvl[b][:, 0]
        otens[dpes.bivl[b]:dpes.bfvl[b]] = out[:, 0]
        if floss:
            predf, truef = daev.cal_dEdxyz_vl(dpes, -out, b, bpath)
            fotens[dpes.bivl[b]:dpes.bfvl[b]] = predf
            fdtens[dpes.bivl[b]:dpes.bfvl[b]] = truef #dpes.fbvl[b]
        if nn_pes.net.arch != 'hfonly':
            otens_lf[dpes.bivl_lf[b]:dpes.bfvl_lf[b]] = out_lf[:, 0]
            dtens_lf[dpes.bivl_lf[b]:dpes.bfvl_lf[b]] = dpes.ybvl_lf[b][:, 0]
            if floss:
                predf_lf, truef_lf = daev.cal_dEdxyz_vl_lf(dpes, -out_lf, b)
                fotens_lf[dpes.bivl_lf[b]:dpes.bfvl_lf[b]] = predf_lf
                fdtens_lf[dpes.bivl_lf[b]:dpes.bfvl_lf[b]] = truef_lf #dpes.fbvl_lf[b]

    if floss:
        #fot = torch.stack(fotens)
        #fdt = torch.stack(fdtens)
        
        fot = list(itertools.chain.from_iterable(fotens))
        fdt = list(itertools.chain.from_iterable(fdtens))
        fot2 = torch.stack(fot)
        fdt2 = torch.stack(fdt)

        #ffnts = fot.shape[0] * fot.shape[1]
        #fot2 = fot.reshape(1, ffnts)
        #fdt2 = fdt.reshape(1, ffnts)

        f_difs = fdt2-fot2
        f_avg_abs_e = torch.mean(abs(f_difs))
        f_rmse = torch.sqrt(torch.mean(f_difs**2))
        f_max_abs_e = torch.max(abs(f_difs))

        #ffnvl = fot.shape[0] * fot.shape[1]
        #fot2 = fot.reshape(1, ffnvl)
        #fdt2 = fdt.reshape(1, ffnvl)
        
        #f_rmse = torch.dist(fot2, fdt2, 2.) / torch.norm(fdt2)
        #f_avg_abs_e = torch.dist(fot2, fdt2, 1.) / torch.norm(fdt2, p=1.)
        #f_max_abs_e = torch.dist(fot2, fdt2, float("inf")) / torch.norm(fdt2, p=float("inf"))

        if nn_pes.net.arch != 'hfonly':
            
            fot_lf = list(itertools.chain.from_iterable(fotens_lf))
            fdt_lf = list(itertools.chain.from_iterable(fdtens_lf))
            fot2_lf = torch.stack(fot_lf)
            fdt2_lf = torch.stack(fdt_lf)


            f_difs_lf = fdt2_lf-fot2_lf
            f_avg_abs_e_lf = torch.mean(abs(f_difs_lf))
            f_rmse_lf = torch.sqrt(torch.mean(f_difs_lf**2))
            f_max_abs_e_lf = torch.max(abs(f_difs_lf))
            
            #fot_lf = torch.stack(fotens_lf)
            #fdt_lf = torch.stack(fdtens_lf)
            #ffnvl_lf = fot_lf.shape[0] * fot_lf.shape[1]
            #fot2_lf = fot_lf.reshape(1, ffnvl_lf)
            #fdt2_lf = fdt_lf.reshape(1, ffnvl_lf)

            #f_rmse_lf = torch.dist(fot2_lf, fdt2_lf, 2.) / torch.norm(fdt2_lf)
            #f_avg_abs_e_lf = torch.dist(fot2_lf, fdt2_lf, 1.) / torch.norm(fdt2_lf, p=1.)
            #f_max_abs_e_lf = torch.dist(fot2_lf, fdt2_lf, float("inf")) / torch.norm(fdt2_lf, p=float("inf"))

    rmse = torch.dist(otens, dtens, 2.) / math.sqrt(fnvl)
    avg_abs_e = torch.dist(otens, dtens, 1.) / fnvl
    max_abs_e = torch.dist(otens, dtens, float("inf"))
    if nn_pes.net.arch != 'hfonly':
        rmse_lf = torch.dist(otens_lf, dtens_lf, 2.) / math.sqrt(fnvl_lf)
        avg_abs_e_lf = torch.dist(otens_lf, dtens_lf, 1.) / fnvl_lf
        max_abs_e_lf = torch.dist(otens_lf, dtens_lf, float("inf"))

    if nn_pes.net.arch != 'hfonly':
        if floss:
            return [kcpm(rmse.detach()), kcpm(avg_abs_e.detach()), kcpm(max_abs_e.detach()), kcpm(rmse_lf.detach()),
                    kcpm(avg_abs_e_lf.detach()), kcpm(max_abs_e_lf.detach()), kcpm(f_rmse.detach()), kcpm(
                    f_avg_abs_e.detach()), kcpm(f_max_abs_e.detach()), kcpm(f_rmse_lf.detach()),
                    kcpm(f_avg_abs_e_lf.detach()), kcpm(f_max_abs_e_lf.detach())]
        else:
            return [kcpm(rmse.detach()), kcpm(avg_abs_e.detach()), kcpm(max_abs_e.detach()), kcpm(rmse_lf.detach()),
                    kcpm(avg_abs_e_lf.detach()), kcpm(max_abs_e_lf.detach()), *[torch.tensor(0.) for i in range(6)]]
    else:
        if floss:
            return [kcpm(rmse.detach()), kcpm(avg_abs_e.detach()), kcpm(max_abs_e.detach()),
                    *[torch.tensor(0.) for i in range(3)], kcpm(f_rmse.detach()), kcpm(
                    f_avg_abs_e.detach()), kcpm(f_max_abs_e.detach()), *[torch.tensor(0.) for i in range(3)]]
        else:
            return [kcpm(rmse.detach()), kcpm(avg_abs_e.detach()), kcpm(max_abs_e.detach()),
                    *[torch.tensor(0.) for i in range(9)]]

# ====================================================================================================
def zt(a, s):
    if a == 0 and s == 0:
        return 'Z'
    elif a == 0 or s == 0:
        return 'z'
    else:
        return ''


# ====================================================================================================
def train(nn_pes, dpes, criterion, optimizer, args, verbose=False):
    if verbose:
        if not args.new_model:
            net_orig = prep_model(True, args, load_opt=False)

    if args.refinehf:
        for param in nn_pes.net.NNa.parameters():
            param.requires_grad = False
        for param in nn_pes.net.NNb.parameters():
            param.requires_grad = False
    elif args.refinelf:
        for param in nn_pes.net.NN_hf.parameters():
            param.requires_grad = False
        for param in nn_pes.net.NN_hf_b.parameters():
            param.requires_grad = False

    try:
        flog = open(args.flognm, "w")
        if args.savets:
            flog_ts = open(args.flogtsnm, "w")
        if args.savevl:
            flog_vl = open(args.flogvlnm, "w")
        if args.savepr:
            flog_pr = open(args.flogprnm, "w")
    except IOError:
        print("Could not open file: flog.dat")
        sys.exit()

    if args.lrscheduler != None:
        lrscheduler = args.lrscheduler[0]
        if 'auto' not in lrscheduler:
            if lrscheduler == 'exp':
                my_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.decayrate)
            elif lrscheduler == 'step':
                my_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.decayrate, last_epoch=-1)
            elif lrscheduler == 'rop':
                if args.tvt[1] == 0:
                    print('ReduceLROnPlateau LR schecular requires validation set')
                    sys.exit()
                #my_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decayrate, patience=50, threshold=0.05)
                my_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decayrate, patience=25, threshold=0.05, verbose=True)
            print('learning rate scheduler: \n', my_lr_scheduler.__dict__)
    else:
        lrscheduler = None
    flog.write(
        '# 1:Epoch 2:min_bRMSloss 3:avg_bRMSloss 4:max_bRMSloss 5:min_EbRMSloss 6:avg_EbRMSloss 7:max_EbRMSloss 8:min_fbRMSloss 9:avg_fbRMSloss 10:max_fbRMSloss 11:L2 12:L1 13:Linf 14:fL2 15:fL1 16:fLinf 17:L2_lf 18:L1_lf 19:Linf_lf 20:fL2_lf 21:fL1_lf 22:fLinf_lf 23:dt 24:dt1 25:dt2 26:dt3 27-31:dtl 32-33:tg 34-35:tf\n')
    flog.flush()
    if args.savets:
        flog_ts.write(
            '# 1:Epoch 2:L2 3:L1 4:Linf 5:fL2 6:fL1 7:fLinf 8:L2_lf 9:L1_lf 10:Linf_lf 11:fL2_lf 12:fL1_lf 13:fLinf_lf\n')
        flog_ts.flush()
    if args.savevl:
        flog_vl.write(
            '# 1:Epoch 2:L2 3:L1 4:Linf 5:fL2 6:fL1 7:fLinf 8:L2_lf 9:L1_lf 10:Linf_lf 11:fL2_lf 12:fL1_lf 13:fLinf_lf\n')
        flog_vl.flush()
    if args.savepr:
        flog_pr.write('# 1:Epoch ')
        for i in range(dpes.ntsdat):
            flog_pr.write('{}:{} '.format(i+2, dpes.labelbts[0][i]))
        flog_pr.write('\n')
        flog_pr.flush()

    # retain graph flag
    # somehow needs this to be True when bsz=1 AND either some na[.] or nb[.] = 0
    # nb s3.py does not require this strangely
    # based on this:
    # https://discuss.pytorch.org/t/how-to-backward-only-a-subset-of-neural-network-parameters-avoid-retain-graph-true/42799
    # I suspect the local data inside the loop over i in s3.py meant it's ok
    # while the global data access here inside the loop over b leads to this
    # tried some obvious localizing but didn't help
    # something for future consideration to avoid retaining the graph needlessly
    # equivalently, if any xb[.][.].nelement() == 0, set True

    rtgfl = False
    # for b in range(dpes.nbttr):
    #     for j in range(dpes.num_nn):
    #         if sum([len(dpes.xdat[i][j]) for i in range(dpes.bitr[b], dpes.bftr[b])]) == 0:
    #             rtgfl = True
    #             print("Warning: Setting rtgfl True!")
    #             break
    #=========================================================================================
    # IC error

    print("Reporting IC error on training data:")

    epoch = 0
    dpes.tf = np.zeros(2, dtype=np.float64)
    rmse, avg_abs_e, max_abs_e, rmse_lf, avg_abs_e_lf, max_abs_e_lf, \
    f_rmse, f_avg_abs_e, f_max_abs_e, f_rmse_lf, f_avg_abs_e_lf, f_max_abs_e_lf = eval_err_tr(nn_pes, dpes, bpath=args.trpath, floss=args.floss)
    if args.savets:
        ts_rmse, ts_avg_abs_e, ts_max_abs_e, ts_rmse_lf, ts_avg_abs_e_lf, ts_max_abs_e_lf, \
        ts_f_rmse, ts_f_avg_abs_e, ts_f_max_abs_e, \
        ts_f_rmse_lf, ts_f_avg_abs_e_lf, ts_f_max_abs_e_lf = eval_err_ts(nn_pes, dpes, bpath=args.tspath, floss=args.floss, outlf=args.eval_err_lf)
    if args.tvt[1] > 0:
        vl_rmse, vl_avg_abs_e, vl_max_abs_e, vl_rmse_lf, vl_avg_abs_e_lf, vl_max_abs_e_lf, \
        vl_f_rmse, vl_f_avg_abs_e, vl_f_max_abs_e, \
        vl_f_rmse_lf, vl_f_avg_abs_e_lf, vl_f_max_abs_e_lf = eval_err_vl(nn_pes, dpes, bpath=args.vlpath, floss=args.floss)
    if args.savepr:
        pr = nn_pes.eval_ts(dpes, 0)[1].detach()


    print(
        "Epoch {:06d} bRMSloss: {:012.6f} {:012.6f} {:012.6f} EbRMSloss: {:012.6f} {:012.6f} {:012.6f} fbRMSloss: {:012.6f} {:012.6f} {:012.6f} L2: {:012.6f} L1: {:012.6f} Linf: {:012.6f} fL2: {:012.6f} fL1: {:012.6f} fLinf: {:012.6f} L2_lf: {:012.6f} L1_lf: {:012.6f} Linf_lf: {:012.6f} fL2_lf: {:012.6f} fL1_lf: {:012.6f} fLinf_lf: {:012.6f} tym: {:07.2f} sec {:05.2f} {:06.2f} {:05.2f} :".
            format(epoch, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   rmse, avg_abs_e, max_abs_e, f_rmse, f_avg_abs_e, f_max_abs_e,
                   rmse_lf, avg_abs_e_lf, max_abs_e_lf, f_rmse_lf, f_avg_abs_e_lf, f_max_abs_e_lf,
                   0.0, 0.0, 0.0, 0.0), ' '.join('{:06.2f}'.format(d) for d in np.zeros(5, dtype=np.float64)),
        '(', ' '.join('{:05.2f}'.format(d) for d in np.zeros(2, dtype=np.float64)), ') sec')
    flog.write(
        "{:06d} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:09.4f} {:09.4f} {:09.4f} {:09.4f} ".
        format(epoch, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               rmse, avg_abs_e, max_abs_e, f_rmse, f_avg_abs_e, f_max_abs_e,
               rmse_lf, avg_abs_e_lf, max_abs_e_lf, f_rmse_lf, f_avg_abs_e_lf, f_max_abs_e_lf,
               0.0, 0.0, 0.0, 0.0)
        + ' '.join('{:09.4f}'.format(d) for d in np.zeros(5, dtype=np.float64)) + ' '
        + ' '.join('{:09.4f}'.format(d) for d in np.zeros(4, dtype=np.float64))
        + '\n')
    flog.flush()
    if args.savets:
        flog_ts.write(
            "{:06d} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e}".
            format(epoch, ts_rmse, ts_avg_abs_e, ts_max_abs_e, ts_f_rmse, ts_f_avg_abs_e, ts_f_max_abs_e,
                   ts_rmse_lf, ts_avg_abs_e_lf, ts_max_abs_e_lf, ts_f_rmse_lf, ts_f_avg_abs_e_lf, ts_f_max_abs_e_lf) + '\n')
        flog_ts.flush()
    if args.savevl:
        flog_vl.write(
            "{:06d} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e}".
            format(epoch, vl_rmse, vl_avg_abs_e, vl_max_abs_e, vl_f_rmse, vl_f_avg_abs_e, vl_f_max_abs_e,
                   vl_rmse_lf, vl_avg_abs_e_lf, vl_max_abs_e_lf, vl_f_rmse_lf, vl_f_avg_abs_e_lf, vl_f_max_abs_e_lf) + '\n')
        flog_vl.flush()
    if args.savepr:
        flog_pr.write('{:06d} '.format(epoch))
        for i in range(dpes.ntsdat):
            flog_pr.write('{:022.14e} '.format(pr[i][0]))
        flog_pr.write('\n')
        flog_pr.flush()

    # =========================================================================================
    # useful: to reshape any tensor t of arb size to a 1d tensor, do: t.reshape(t.numel())
    # or to a 2d [1xN] tensor: t.reshape([1,t.numel()])
    # or to a 2d [Nx1] tensor: t.reshape([t.numel(),1])
    # t.numel() returns the number of elements in the tensor t

    bat_lst = list(range(dpes.nbttr))

    dt1 = 0.0
    dt2 = 0.0
    dt3 = 0.0
    dt = 0.0
    tl = np.zeros(6, dtype=np.float64)
    dtl = np.zeros(5, dtype=np.float64)
    nn_pes.tg = np.zeros(2, dtype=np.float64)
    dpes.tf = np.zeros(2, dtype=np.float64)

    tloop = timeit.default_timer()
    if not args.new_model:
        if nn_pes.net.arch != 'hfonly':
            minerr = rmse * 0.5 + rmse_lf * 0.5
        else:
            minerr = rmse

        if args.tvt[1] > 0:
            if nn_pes.net.arch != 'hfonly':
                vl_minerr = vl_rmse * 0.5 + vl_rmse_lf * 0.5
            else:
                vl_minerr = vl_rmse

        crt_err = args.crt_err
    else:
        minerr = 100.
        vl_minerr = 100.
        crt_err = 100.

    dEsq = (dpes.ymax - dpes.ymin)**2
    #print('dEsq: ', dEsq)
    if nn_pes.net.arch != 'hfonly':
        dElsq = (dpes.ymax_lf - dpes.ymin_lf) ** 2
    else:
        dElsq = 1.
    if args.floss:
        dfsq = (dpes.fmax - dpes.fmin) ** 2
        print('dpes.fmin: ', dpes.fmin)
        print('dpes.fmax: ', dpes.fmax)
        print('dfsq: ', dfsq)
    if args.grad_trvl:
        args.stop_grad = True
    if lrscheduler == 'autograd' or args.stop_grad:
        n_erravg = int(args.grad_step * 0.1)
        if n_erravg < 100:
            n_erravg = 100
        elif n_erravg >= 1000:
            n_erravg = 1000
        vl_err_prior = np.zeros(n_erravg)
        vl_err_cur = np.zeros(n_erravg)
        tr_err_prior = np.zeros(n_erravg)
        tr_err_cur = np.zeros(n_erravg)
        changed_lr = False

    #================================================================================================
    """
    If performing force training, this sets up the class to handle the relative weights of the force and energy terms.
    The class can handle more than 2 tasks by changing num_tasks and passing additional loss terms during training.
    """
    if args.floss:
        if args.optimize_force_weight:
            #if not args.new_model:
            #    log_sigma = torch.nn.Parameter(nn_pes.net.log_sigma)
            #else:
            #    log_sigma = None
            log_sigma = None
            mtl = task_weights(num_tasks=2, log_sigma=log_sigma, traditional=False).to(args.device)
            nn_pes.net.log_sigma = torch.nn.Parameter(mtl.log_sigma)
            optimizer.param_groups[0]['params'].append(mtl.log_sigma)
            print('Force weight optimization is ON!')
        else:
            log_sigma = torch.tensor([args.fw], requires_grad=True)
            mtl = task_weights(num_tasks=2, log_sigma=log_sigma, traditional=True).to(args.device)
    #================================================================================================

    
    #myloss = lossfunction(args.wloss, args.floss).lossfn          #unused?

    while epoch != args.epochs:
        epoch = epoch + 1
        t0 = timeit.default_timer()

        if batch_list_shuffle:
            random.shuffle(bat_lst)
        t1 = timeit.default_timer()

        epred = torch.empty(dpes.ntrdat, device=dpes.device)
        edata = torch.empty(dpes.ntrdat, device=dpes.device)
        fpred = [[] for b in range(dpes.nbttr)]
        fdata = [[] for b in range(dpes.nbttr)]

        aloss = torch.zeros(dpes.nbttr)
        eloss = torch.zeros(dpes.nbttr)
        floss = torch.zeros(dpes.nbttr)
        for b in bat_lst:
            optimizer.zero_grad()
            tl[0] = timeit.default_timer()

            ylf, output = nn_pes.eval_tr(dpes, b)
            tl[1] = timeit.default_timer()

            if args.floss:
                predf, truef = daev.cal_dEdxyz_tr(dpes, -output, b, bpath=args.trpath)
                if nn_pes.net.arch != 'hfonly':
                    predfl, truefl = daev.cal_dEdxyz_tr_lf(dpes, -ylf, b)
            tl[2] = timeit.default_timer()

            # optimizer.zero_grad()
            # ylf, output = nn_pes.eval_tr(dpes, b)
            if args.wloss:
                if args.floss:
                    if nn_pes.net.arch != 'hfonly':
                        ls_e, ls_f = my_loss_wf(args.fw, dpes.wbtr[b], output, dpes.ybtr[b], predf,
                                                      dpes.fbtr[b], ylf, dpes.ybtr_lf[b], predfl, dpes.fbtr_lf[b], dEsq, dElsq, dfsq)
                        #floss[b] = ls_f
                        #eloss[b] = ls_e
                        #aloss[b] = loss.detach()
                    else:
                        ls_e, ls_f = my_loss_wf(args.fw, dpes.wbtr[b], output, dpes.ybtr[b], predf,
                                                      dpes.fbtr[b], None, None, None, None, dEsq, dElsq, dfsq)
                        #floss[b] = ls_f
                        #eloss[b] = ls_e
                        #aloss[b] = loss.detach()
                    floss[b] = ls_f
                    eloss[b] = ls_e
                    loss = mtl(ls_e, ls_f)
                    aloss[b] = loss
                else:
                    if nn_pes.net.arch != 'hfonly':
                        eloss[b] = criterion(output, dpes.ybtr[b], ylf, dpes.ybtr_lf[b]).detach()
                        loss = my_loss_w(dpes.wbtr[b], output, dpes.ybtr[b], ylf, dpes.ybtr_lf[b])
                        aloss[b] = loss.detach()
                    else:
                        eloss[b] = criterion(output, dpes.ybtr[b], None, None).detach()
                        loss = my_loss_w(dpes.wbtr[b], output, dpes.ybtr[b], None, None)
                        aloss[b] = loss.detach()
            else:
                if args.floss:
                    if nn_pes.net.arch != 'hfonly':
                        ls_e, ls_f = my_loss_f(args.fw, output, dpes.ybtr[b], predf, dpes.fbtr[b], dpes.pfdims_tr[b], ylf,
                                                      dpes.ybtr_lf[b], predfl, dpes.fbtr_lf[b], dEsq=1.0, dElsq=1.0, dfsq=1.0, c=args.costratio)
                        #floss[b] = ls_f
                        #eloss[b] = ls_e
                        #aloss[b] = loss.detach()
                    else:
                        #ls_e, ls_f = my_loss_f_rel(args.fw, output, dpes.ybtr[b], predf, truef, dpes.pfdims_tr[b], dpes.expw[b], None, None,
                        #                              None, None, dEsq=dEsq, dElsq=1.0, dfsq=dfsq)
                        ls_e, ls_f = my_loss_f(args.fw, output, dpes.ybtr[b], predf, truef, dpes.pfdims_tr[b], None, None,
                                                      None, None, dEsq=dEsq, dElsq=1.0, dfsq=dfsq)
                        #floss[b] = ls_f
                        #eloss[b] = ls_e
                        #aloss[b] = loss.detach()
                    floss[b] = ls_f
                    eloss[b] = ls_e
                    loss = mtl(ls_e, ls_f)
                    aloss[b] = loss
                else:
                    if nn_pes.net.arch == 'hfonly' or args.refinehf:
                        loss = my_loss(output, dpes.ybtr[b], None, None, dEsq=1.0)
                    else:
                        if args.refinelf:
                            loss = my_loss(output*0., dpes.ybtr[b]*0., ylf, dpes.ybtr_lf[b], dEsq, dElsq, c=args.costratio)
                        else:
                            loss = my_loss(output, dpes.ybtr[b], ylf, dpes.ybtr_lf[b], dEsq, dElsq, c=args.costratio)
                    eloss[b] = loss.detach()
                    aloss[b] = eloss[b]
            tl[3] = timeit.default_timer()
            
            loss.backward(retain_graph=rtgfl)
            loss.retain_grad() 
            tl[4] = timeit.default_timer()

            edata[dpes.bitr[b]:dpes.bftr[b]] = dpes.ybtr[b][:, 0]
            epred[dpes.bitr[b]:dpes.bftr[b]] = output[:, 0].detach()
            if args.floss:
                fpred[dpes.bitr[b]:dpes.bftr[b]] = [fmat.detach() for fmat in predf]
                fdata[dpes.bitr[b]:dpes.bftr[b]] = truef#dpes.fbtr[b]


            optimizer.step()
            tl[5] = timeit.default_timer()

            for i in range(5):
                dtl[i] += tl[i + 1] - tl[i]

            if lrscheduler == 'exp' or lrscheduler == 'step':
                my_lr_scheduler.step()
        
        
        t2 = timeit.default_timer()

        rmse = kcpm(torch.dist(epred, edata, 2.) / np.sqrt(float(dpes.ntrdat)))
        avg_abs_e = kcpm(torch.dist(epred, edata, 1.) / float(dpes.ntrdat))
        max_abs_e = kcpm(torch.dist(epred, edata, float("inf")))
        if args.floss:
            fot = torch.stack(fpred)
            fdt = torch.stack(fdata)
            #fot = list(itertools.chain.from_iterable(fot))
            #fdt = list(itertools.chain.from_iterable(fdt))
            #fot2 = torch.stack(fot)
            #fdt2 = torch.stack(fdt)


            #f_difs = fdt2-fot2
            #f_avg_abs_e = torch.mean(abs(f_difs))
            #f_rmse = torch.sqrt(torch.mean(f_difs**2))
            #f_max_abs_e = torch.max(abs(f_difs))
            ffntr = fot.shape[0] * fot.shape[1]
            fot2 = fot.reshape(1, ffntr)
            fdt2 = fdt.reshape(1, ffntr)
            f_rmse = kcpm(torch.dist(fot2, fdt2, 2.) / np.sqrt(ffntr))
            f_avg_abs_e = kcpm(torch.dist(fot2, fdt2, 1.) / ffntr)
            f_max_abs_e = kcpm(torch.dist(fot2, fdt2, float("inf")))
        # print(rmse, avg_abs_e, max_abs_e, f_rmse, f_avg_abs_e, f_max_abs_e)
        if epoch % args.ntlp == 0 or epoch == args.epochs:
            rmse, avg_abs_e, max_abs_e, rmse_lf, avg_abs_e_lf, max_abs_e_lf, \
            f_rmse, f_avg_abs_e, f_max_abs_e, f_rmse_lf, f_avg_abs_e_lf, f_max_abs_e_lf = eval_err_tr(nn_pes, dpes, bpath=args.trpath, floss=args.floss)
            if args.savets:
                ts_rmse, ts_avg_abs_e, ts_max_abs_e, ts_rmse_lf, ts_avg_abs_e_lf, ts_max_abs_e_lf, \
                ts_f_rmse, ts_f_avg_abs_e, ts_f_max_abs_e, \
                ts_f_rmse_lf, ts_f_avg_abs_e_lf, ts_f_max_abs_e_lf = eval_err_ts(nn_pes, dpes, bpath=args.tspath, floss=args.floss, outlf=args.eval_err_lf)
            if args.savepr:
                pr = nn_pes.eval_ts(dpes, 0)[1].detach()

        if args.tvt[1] > 0:
            vl_rmse, vl_avg_abs_e, vl_max_abs_e, vl_rmse_lf, vl_avg_abs_e_lf, vl_max_abs_e_lf, \
            vl_f_rmse, vl_f_avg_abs_e, vl_f_max_abs_e, \
            vl_f_rmse_lf, vl_f_avg_abs_e_lf, vl_f_max_abs_e_lf = eval_err_vl(nn_pes, dpes, bpath=args.vlpath,  floss=args.floss)

        if lrscheduler == 'rop':
            my_lr_scheduler.step(vl_rmse)

        
        t3 = timeit.default_timer()

        dt1 += t1 - t0
        dt2 += t2 - t1
        dt3 += t3 - t2
        dt += t2 - t0 + dt3

        if verbose:
            if nn_pes.net.arch != 'hfonly':
                test_equal_hf = [[], []]
                for p1, p2 in zip(net_orig.net.NN_hf.parameters(), nn_pes.net.NN_hf.parameters()):
                    if p1.data.ne(p2.data).sum() > 0:
                        test_equal_hf[0].append(1)
                    else:
                        test_equal_hf[0].append(0)
                for p1, p2 in zip(net_orig.net.NN_hf_b.parameters(), nn_pes.net.NN_hf_b.parameters()):
                    if p1.data.ne(p2.data).sum() > 0:
                        test_equal_hf[1].append(1)
                    else:
                        test_equal_hf[1].append(0)

                test_equal_grad_hf = [[], []]
                for pars in nn_pes.net.NN_hf.parameters():
                    grad = pars.grad
                    test_equal_grad_hf[0].append(grad.sum().detach().tolist())


                for pars in nn_pes.net.NN_hf_b.parameters():
                    grad = pars.grad
                    test_equal_grad_hf[1].append(grad.sum().detach().tolist())

                if np.sum(test_equal_hf) > 0:
                    print('NN_hf_a and NN_hf_b parameters were changed.', test_equal_hf)
                else:
                    print('NN_hf_a and NN_hf_b parameters were not changed.')

                print('NN_hf_a and NN_hf_b weight gradients: \n', end='')
                for iii in range(len(test_equal_grad_hf)):
                    print('NN_hf_{}:'.format(iii), test_equal_grad_hf[iii])


                print('NNa parameters (require grad): ', end='')
                for param in nn_pes.net.NN_hf.parameters():
                    print(param.requires_grad, end=' ')
                print('\nNN_hf parameters (require grad): ', end='')
                for param in nn_pes.net.NN_hf.parameters():
                    print(param.requires_grad, end=' ')
                print('')

        aloss = torch.sqrt(aloss.detach())
        eloss = torch.sqrt(eloss.detach())
        floss = torch.sqrt(floss.detach())
        if epoch % args.ntlp == 0 or epoch == args.epochs:
            if args.optimize_force_weight:
                print('task weight precisions: ', mtl.precisions)
                print('task weight sigmas: ', mtl.log_sigma)
            print(
                "Epoch {:06d} bRMSloss: {:012.6f} {:012.6f} {:012.6f} EbRMSloss: {:012.6f} {:012.6f} {:012.6f} fbRMSloss: {:012.6f} {:012.6f} {:012.6f} L2: {:012.6f} L1: {:012.6f} Linf: {:012.6f} fL2: {:012.6f} fL1: {:012.6f} fLinf: {:012.6f} L2_lf: {:012.6f} L1_lf: {:012.6f} Linf_lf: {:012.6f} fL2_lf: {:012.6f} fL1_lf: {:012.6f} fLinf_lf: {:012.6f} tym: {:07.2f} sec {:05.2f} {:06.2f} {:05.2f} :".
                    format(epoch, torch.min(aloss), torch.mean(aloss), torch.max(aloss),
                           torch.min(eloss), torch.mean(eloss), torch.max(eloss),
                           torch.min(floss), torch.mean(floss), torch.max(floss),
                           rmse, avg_abs_e, max_abs_e, f_rmse, f_avg_abs_e, f_max_abs_e,
                           rmse_lf, avg_abs_e_lf, max_abs_e_lf, f_rmse_lf, f_avg_abs_e_lf, f_max_abs_e_lf,
                           dt, dt1, dt2, dt3), ' '.join('{:06.2f}'.format(d) for d in np.zeros(5, dtype=np.float64)),
                ' '.join('{:06.2f}'.format(d) for d in dtl),
                '(', ' '.join('{:05.2f}'.format(d) for d in nn_pes.tg), ':',
                '(', ' '.join('{:05.2f}'.format(d) for d in dpes.tf), ') sec')
            flog.write(
                "{:06d} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:09.4f} {:09.4f} {:09.4f} {:09.4f} ".
                format(epoch, torch.min(aloss), torch.mean(aloss), torch.max(aloss),
                           torch.min(eloss), torch.mean(eloss), torch.max(eloss),
                           torch.min(floss), torch.mean(floss), torch.max(floss),
                           rmse, avg_abs_e, max_abs_e, f_rmse, f_avg_abs_e, f_max_abs_e,
                           rmse_lf, avg_abs_e_lf, max_abs_e_lf, f_rmse_lf, f_avg_abs_e_lf, f_max_abs_e_lf,
                           dt, dt1, dt2, dt3)
                + ' '.join('{:09.4f}'.format(d) for d in dtl) + ' '
                + ' '.join('{:09.4f}'.format(d) for d in nn_pes.tg) + ' '
                + ' '.join('{:09.4f}'.format(d) for d in dpes.tf)
                + '\n')
            flog.flush()
            if args.savets:
                flog_ts.write(
                    "{:06d} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e}".
                    format(epoch, ts_rmse, ts_avg_abs_e, ts_max_abs_e, ts_f_rmse, ts_f_avg_abs_e, ts_f_max_abs_e,
                           ts_rmse_lf, ts_avg_abs_e_lf, ts_max_abs_e_lf, ts_f_rmse_lf, ts_f_avg_abs_e_lf,
                           ts_f_max_abs_e_lf) + '\n')
                flog_ts.flush()
            if args.savevl:
                flog_vl.write(
                    "{:06d} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e}".
                    format(epoch, vl_rmse, vl_avg_abs_e, vl_max_abs_e, vl_f_rmse, vl_f_avg_abs_e, vl_f_max_abs_e,
                           vl_rmse_lf, vl_avg_abs_e_lf, vl_max_abs_e_lf, vl_f_rmse_lf, vl_f_avg_abs_e_lf,
                           vl_f_max_abs_e_lf) + '\n')
                flog_vl.flush()
            if args.savepr:
                flog_pr.write('{:06d} '.format(epoch))
                for i in range(dpes.ntsdat):
                    flog_pr.write('{:022.14e} '.format(pr[i][0]))
                flog_pr.write('\n')
                flog_pr.flush()
            dt1 = 0.0
            dt2 = 0.0
            dt3 = 0.0
            dt = 0.0
            dtl = np.zeros(5, dtype=np.float64)
            dpes.tf = np.zeros(2, dtype=np.float64)
            nn_pes.tg = np.zeros(2, dtype=np.float64)
            if printlr:
                if epoch == args.ntlp:
                    lrlist = []
                if lrscheduler != None:
                    if lrscheduler != 'auto':
                        lrlist.append([epoch, my_lr_scheduler.get_lr()[0], rmse.detach()])
                    else:
                        lrlist.append([epoch, optimizer.param_groups[0]['lr'], rmse.detach()])
                else:
                    lrlist.append([epoch, args.learning_rate, rmse.detach()])


        if not args.nw and epoch % args.save_every == 0:
            nn_pes.save(args, args.savepth + args.savepth_pars + "comp", epoch, optimizer, crt_err)

        if nn_pes.net.arch != 'hfonly':
            if args.refinelf:
                rmse_total = rmse_lf
            else:
                rmse_total = rmse * 0.5 + rmse_lf * 0.5
        else:
            rmse_total = rmse

        if args.tvt[1] > 0:
            if nn_pes.net.arch != 'hfonly':
                if args.refinelf:
                    vl_rmse_total = vl_rmse_lf
                else:
                    vl_rmse_total = vl_rmse * 0.5 + vl_rmse_lf * 0.5
            else:
                vl_rmse_total = vl_rmse
            if not args.nw and vl_minerr > vl_rmse_total:
                nn_pes.save(args, args.savepth + args.savepth_pars + "/min_vl/comp", 0, optimizer, crt_err)
                vl_minerr = vl_rmse_total

        if not args.nw and minerr > rmse_total:
            if args.optimizer[0] != 'SGD':
                nn_pes.save(args, args.savepth + args.savepth_pars + "/min/comp", 0, optimizer, crt_err)
                minerr = rmse_total


        if args.stop_error != None and args.stop_error > rmse_total:
            print('error was reached to {}        >>        stop training'.format(rmse_total))
            break

        if lrscheduler == 'autograd' or args.stop_grad:
            if dpes.nvldat == 0:
                print('validation set is required to set \'autograd\' learning rate scheduler or early-termination condition')
                sys.exit()
            if epoch < args.grad_step:
                mm = epoch % args.grad_step
                if mm in range(1, n_erravg+1):
                    vl_err_prior[mm - 1] = vl_rmse_total
                    tr_err_prior[mm - 1] = rmse_total
                if mm in range(args.grad_step-n_erravg, args.grad_step):
                    vl_err_cur[mm - (args.grad_step - n_erravg)] = vl_rmse_total
                    tr_err_cur[mm - (args.grad_step - n_erravg)] = rmse_total
            else:
                mm = epoch % args.grad_step
                if mm in range(args.grad_step-n_erravg, args.grad_step):
                    vl_err_cur[mm - (args.grad_step - n_erravg)] = vl_rmse_total
                    tr_err_cur[mm - (args.grad_step - n_erravg)] = rmse_total

            if mm == 0 and epoch > 1:
                vl_err_diff = np.mean(vl_err_cur) - np.mean(vl_err_prior)
                tr_err_diff = np.mean(tr_err_cur) - np.mean(tr_err_prior)
                # vl_err_diff = np.mean(vl_err_cur)/np.mean(vl_err_prior)
                print('epoch: {}\ncurrent mean error - tr:{}, vl:{}\nprevious mean error - tr:{}, vl:{}\nerror diff - tr:{}, vl:{}'.format(epoch, np.mean(tr_err_cur), np.mean(vl_err_cur), np.mean(tr_err_prior), np.mean(vl_err_prior), tr_err_diff, vl_err_diff))

                # if vl_err_diff > 0.99 and rmse < 1: #if abs(vl_err_diff) < 0.01 and rmse < 1:
                #     print('error was stuck around {}         >>         stop training'.format(rmse_total))
                #     nn_pes.save(args, args.savepth + args.savepth_pars + "/min/comp", 0, optimizer)
                #     break
                increasing_vl_err = False
                if vl_err_diff > 0:
                    if args.grad_trvl:
                        if tr_err_diff > 0:
                            print(
                                'validation error increases ({}) and training error increases ({})'.format(
                                    vl_err_diff, tr_err_diff))
                        else:
                            if abs(vl_err_diff) > abs(tr_err_diff):
                                increasing_vl_err = True
                                print('increase in validation error ({}) >> decrease in training error ({})'.format(vl_err_diff, tr_err_diff))
                            else:
                                print(
                                    'increase in validation error ({}) ~= decrease in training error ({})'.format(
                                        vl_err_diff, tr_err_diff))
                    else:
                        increasing_vl_err = True
                if increasing_vl_err:
                    if lrscheduler == 'autograd':
                        if not changed_lr:
                            for param_group in optimizer.param_groups:
                                newlr = param_group['lr'] * args.decayrate
                                if newlr >= 1e-6:
                                    param_group['lr'] = newlr
                                    changed_lr = True
                            if changed_lr == True:
                                print('validation error is increasing        >>        learning rate is changed to ',
                                      newlr)
                            else:
                                if args.stop_grad:
                                    if args.stop_grad_err is not None:
                                        if rmse > args.stop_grad_err:
                                            print(
                                                'validation error is increasing but training error is {} (> {} kcal/mol).'.format(
                                                    rmse, args.stop_grad_err))
                                        else:
                                            print(
                                                'validation error is increasing and training error is {} (< {} kcal/mol)       >>        stop training'.format(
                                                    rmse, args.stop_grad_err))
                                            break
                                    else:
                                        print('validation error is increasing        >>        stop training')
                                        break
                        else:
                            if args.stop_grad_err is not None:
                                if rmse > args.stop_grad_err:
                                    print('validation error is increasing but training error is {} (> {} kcal/mol).'.format(rmse, args.stop_grad_err))
                                else:
                                    print('validation error is increasing and training error is {} (< {} kcal/mol)       >>        stop training'.format(rmse, args.stop_grad_err))
                                    break
                            else:
                                print('validation error is increasing        >>        stop training')
                                break

                    else:
                        if args.stop_grad_err is not None:
                            if rmse > args.stop_grad_err:
                                print('error is increasing but training error is {} which is greater than {} kcal/mol.'.format(rmse, args.stop_grad_err))
                            else:
                                print(
                                    'validation error is increasing and training error is {} (< 1 kcal/mol)       >>        stop training'.format(
                                        rmse))
                                break
                        else:
                            print('validation error is increasing        >>        stop training')
                            break

                else:
                    changed_lr = False
                vl_err_prior = vl_err_cur.copy()
                tr_err_prior = tr_err_cur.copy()

        if lrscheduler == 'auto':
            if rmse < crt_err:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * args.decayrate
                if crt_err < 1.:
                    crt_err = crt_err * 0.5
                    args.decayrate = 0.5
                else:
                    crt_err = crt_err * 0.1
                # print(lrscheduler)
                print('lrscheduler: ({}), learning rate is changed to {}'.format(lrscheduler, param_group['lr']))


    flog.write(
        "{:06d} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:09.4f} {:09.4f} {:09.4f} {:09.4f} ".
        format(epoch, torch.min(aloss), torch.mean(aloss), torch.max(aloss),
               torch.min(eloss), torch.mean(eloss), torch.max(eloss),
               torch.min(floss), torch.mean(floss), torch.max(floss),
               rmse, avg_abs_e, max_abs_e, f_rmse, f_avg_abs_e, f_max_abs_e,
               rmse_lf, avg_abs_e_lf, max_abs_e_lf, f_rmse_lf, f_avg_abs_e_lf, f_max_abs_e_lf,
               dt, dt1, dt2, dt3)
        + ' '.join('{:09.4f}'.format(d) for d in dtl) + ' '
        + ' '.join('{:09.4f}'.format(d) for d in nn_pes.tg) + ' '
        + ' '.join('{:09.4f}'.format(d) for d in dpes.tf)
        + '\n')
    flog.close()
    if args.savets:
        flog_ts.write(
            "{:06d} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e}".
            format(epoch, ts_rmse, ts_avg_abs_e, ts_max_abs_e, ts_f_rmse, ts_f_avg_abs_e, ts_f_max_abs_e,
                   ts_rmse_lf, ts_avg_abs_e_lf, ts_max_abs_e_lf, ts_f_rmse_lf, ts_f_avg_abs_e_lf,
                   ts_f_max_abs_e_lf) + '\n')
        flog_ts.close()
    if args.savevl:
        flog_vl.write(
            "{:06d} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e} {:022.14e}".
            format(epoch, vl_rmse, vl_avg_abs_e, vl_max_abs_e, vl_f_rmse, vl_f_avg_abs_e, vl_f_max_abs_e,
                   vl_rmse_lf, vl_avg_abs_e_lf, vl_max_abs_e_lf, vl_f_rmse_lf, vl_f_avg_abs_e_lf,
                   vl_f_max_abs_e_lf) + '\n')
        flog_vl.close()
    if args.savepr:
        flog_pr.write('{:06d} '.format(epoch))
        for i in range(dpes.ntsdat):
            flog_pr.write('{:022.14e} '.format(pr[i][0]))
        flog_pr.write('\n')
        flog_pr.close()

    print(
        "Epoch {:06d} bRMSloss: {:012.6f} {:012.6f} {:012.6f} EbRMSloss: {:012.6f} {:012.6f} {:012.6f} fbRMSloss: {:012.6f} {:012.6f} {:012.6f} L2: {:012.6f} L1: {:012.6f} Linf: {:012.6f} fL2: {:012.6f} fL1: {:012.6f} fLinf: {:012.6f} L2_lf: {:012.6f} L1_lf: {:012.6f} Linf_lf: {:012.6f} fL2_lf: {:012.6f} fL1_lf: {:012.6f} fLinf_lf: {:012.6f} tym: {:07.2f} sec {:05.2f} {:06.2f} {:05.2f} :".
            format(epoch, torch.min(aloss), torch.mean(aloss), torch.max(aloss),
                   torch.min(eloss), torch.mean(eloss), torch.max(eloss),
                   torch.min(floss), torch.mean(floss), torch.max(floss),
                   rmse, avg_abs_e, max_abs_e, f_rmse, f_avg_abs_e, f_max_abs_e,
                   rmse_lf, avg_abs_e_lf, max_abs_e_lf, f_rmse_lf, f_avg_abs_e_lf, f_max_abs_e_lf,
                   dt, dt1, dt2, dt3), ' '.join('{:06.2f}'.format(d) for d in np.zeros(5, dtype=np.float64)),
        ' '.join('{:06.2f}'.format(d) for d in dtl),
        '(', ' '.join('{:05.2f}'.format(d) for d in nn_pes.tg), ':',
        '(', ' '.join('{:05.2f}'.format(d) for d in dpes.tf), ') sec')

    dtloop = timeit.default_timer() - tloop
    print("dtloop:", dtloop, "sec")



    if not args.nw:
        print("Saving NNs and optimizer..", end='')

        nn_pes.save(args, args.savepth + args.savepth_pars + "comp", epoch, optimizer, crt_err)
        nn_pes.print_model(args.savepth + args.savepth_pars + "comp", epoch)
        print_opt(optimizer, args.savepth + args.savepth_pars + "optm", epoch)
        print('done')

    if printlr:
        import pandas as pd
        df = pd.DataFrame(lrlist)
        df.columns = ['epoch', 'lr', 'L2']
        df.to_csv('lrlist.csv', index=False)

    if verbose:
        if not args.new_model:
            test_equal = [[], []]
            for p1, p2 in zip(net_orig.net.NNa.parameters(), nn_pes.net.NNa.parameters()):
                if p1.data.ne(p2.data).sum() > 0:
                    test_equal[0].append(1)
                else:
                    test_equal[0].append(0)
            for p1, p2 in zip(net_orig.net.NNb.parameters(), nn_pes.net.NNb.parameters()):
                if p1.data.ne(p2.data).sum() > 0:
                    test_equal[1].append(1)
                else:
                    test_equal[1].append(0)

            if np.sum(test_equal) == 0:
                print('NNa and NNb net parameters were not changed.')
            else:
                print('NNa and NNb net parameters were changed:', test_equal)

    return 0


# ====================================================================================================
def kcpm(E_eV):
    """
    converts energy in Hartree atomic units to kcal/mol
    converts energy in eV to kcal/mol
    """
    # return E_Hartree * Hartree * mol / kcal
    return E_eV * mol / kcal

# ====================================================================================================
def validate(nn_pes, dpes, criterion):
    for b in range(dpes.nbtvl):
        ylf, output = nn_pes.eval_vl(dpes, b)
        if nn_pes.net.arch != 'hfonly':
            loss = criterion(output, dpes.ybvl[b], ylf, dpes.ybvl_lf[b])
        else:
            loss = criterion(output, dpes.ybvl[b], None, None)
        print("Batch {} - Validation RMS loss: {:012.4f} kcal/mol".format(b, kcpm(math.sqrt(loss))))

    return 0


# ====================================================================================================
def output_pred(nn_pes, dpes, outlf=False):
    if dpes.ntsdat > 0:
        otens = torch.empty(dpes.ntsdat, device=dpes.device)
        dtens = torch.empty(dpes.ntsdat, device=dpes.device)
        if nn_pes.net.arch != 'hfonly' and outlf:
            otens_lf = torch.empty(dpes.ntsdat_lf, device=dpes.device)
            dtens_lf = torch.empty(dpes.ntsdat_lf, device=dpes.device)

        for b in range(dpes.nbtts):
            out_lf, out = nn_pes.eval_ts(dpes, b)
            dtens[dpes.bits[b]:dpes.bfts[b]] = kcpm(dpes.ybts[b][:, 0])
            if nn_pes.net.arch != 'hfonly' and outlf:
                otens[dpes.bits[b]:dpes.bfts[b]] = kcpm(out[:, 0])
                otens_lf[dpes.bits_lf[b]:dpes.bfts_lf[b]] = kcpm(out_lf[:, 0])
                dtens_lf[dpes.bits_lf[b]:dpes.bfts_lf[b]] = kcpm(dpes.ybts_lf[b][:, 0])
            else:
                otens[dpes.bits[b]:dpes.bfts[b]] = kcpm(out[:, 0])

    if dpes.ntrdat > 0:
        otens = torch.empty(dpes.ntrdat, device=dpes.device)
        dtens = torch.empty(dpes.ntrdat, device=dpes.device)
        if nn_pes.net.arch != 'hfonly' and outlf:
            otens_lf = torch.empty(dpes.ntrdat_lf, device=dpes.device)
            dtens_lf = torch.empty(dpes.ntrdat_lf, device=dpes.device)

        for b in range(dpes.nbttr):
            out_lf, out = nn_pes.eval_tr(dpes, b)
            dtens[dpes.bitr[b]:dpes.bftr[b]] = kcpm(dpes.ybtr[b][:, 0])

            if nn_pes.net.arch != 'hfonly' and outlf:
                otens[dpes.bitr[b]:dpes.bftr[b]] = kcpm(out[:, 0])
                otens_lf[dpes.bitr_lf[b]:dpes.bftr_lf[b]] = kcpm(out_lf[:, 0])
                dtens_lf[dpes.bitr_lf[b]:dpes.bftr_lf[b]] = kcpm(dpes.ybtr_lf[b][:, 0])

            else:
                otens[dpes.bitr[b]:dpes.bftr[b]] = kcpm(out[:, 0])

    return 0


# ====================================================================================================
def test(nn_pes, dpes, criterion, outlf=False):
    ntssqrt = torch.sqrt(torch.tensor(float(dpes.ntsdat)))
    otens = torch.empty(0, device=dpes.device)
    dtens = torch.empty(0, device=dpes.device)
    for b in range(dpes.nbtts):
        ylf, output = nn_pes.eval_ts(dpes, b)
        if nn_pes.net.arch != 'hfonly' and outlf:
            loss = criterion(output, dpes.ybts[b], ylf, dpes.ybts_lf[b])
        else:
            loss = criterion(output, dpes.ybts[b], None, None)
        otens = torch.cat((otens, kcpm(output)))
        dtens = torch.cat((dtens, kcpm(dpes.ybts[b])))
        print("Batch {} - Testing RMS loss: {:012.4f} kcal/mol".format(b, kcpm(math.sqrt(loss))))
    print("RMS loss: {:012.4f} kcal/mol - avL2: {:012.4f}, avL1: {:012.4f}, Linf: {:012.4f}".
          format(kcpm(math.sqrt(loss)), torch.dist(otens, dtens, 2.) / ntssqrt,
                 torch.dist(otens, dtens, 1.) / float(dpes.ntsdat), torch.dist(otens, dtens, float("inf"))
                 ))

    return 0


# ====================================================================================================
def parse_arguments_list():
    parser = argparse.ArgumentParser(description='PES code')
    parser.add_argument('-g', '--enable-cuda', action='store_true', help='Enable CUDA -- use gpu if available')
    parser.add_argument('-c', '--num_cores', type=int, default=4,
                        help='specify number of cores to run on [4], nb. num_threads=2 x num_cores')
    parser.add_argument('-d', '--input-data-type', choices=('xyz', 'aev', 'pca', 'sqlite'), default='aev',
                        help='specify input data file type [aev]')
    parser.add_argument('-f', '--input-data-fname', nargs='*', default=['aev_db_new.hdf5'],
                        help='Specify input data filename(s) [aev_db.hdf5]')
    parser.add_argument('-n', '--new-model', action='store_true', default=False,
                        help='specify new model, do not load exising net0.pt, net1.pt, opt.pt')
    parser.add_argument('-I', '--netparam-init-amp', type=float, default=1.0,
                        help='specify amplification factor for weights/biases U(-a,a) initialization [1.0]')
    parser.add_argument('-T', '--truncation-fraction', type=float, default=1.0,
                        help='specify decimal fraction (0.0,1.0] by which to truncate data')
    parser.add_argument('-u', '--dont-shuffle-data', action='store_true', default=False,
                        help='do not randomly shuffle data')
    parser.add_argument('-l', '--learning-rate', type=float, default=1.e-3,
                        help='specify optimizer learning rate [1.e-3]')
    parser.add_argument('-m', '--momentum', type=float, default=0.5, help='specify optimizer momentum [0.5]')
    parser.add_argument('--weight-decay', type=float, default=0, help='specify L2-penalty for regularization')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='specify number of epochs for training')
    parser.add_argument('-p', '--ntlp', type=int, default=10,
                        help='specify eval/output training error every <ntlp> epochs')
    parser.add_argument('--save-every', type=int, default=10,
                        help='specify save NN every <save-every> epochs')
    parser.add_argument('-b', '--tr_batch_size', type=int, default=8,
                        help='specify (approximate) batch size for training')
    parser.add_argument('--ni', type=int, default=0,
                        help='ni is the starting data sample index, ni \in [0,ndat-1], default 0')
    parser.add_argument('--nf', type=int, default=None,
                        help='nf-1 is the ending data sample index, nf \in [1,ndat], default None (=>ndat whatever it is)')
    parser.add_argument('--na', action='store_true', default=False, help='do not write _act.hdf5 file')
    parser.add_argument('--nw', action='store_true', default=False, help='do not write net & opt files')
    parser.add_argument('--no_output_pred', action='store_true', default=False, help='do not call output_pred')
    parser.add_argument('-nt', '--nntype', nargs='*', default=['Comp'], help='Specify net type [Net or Comp]')
    parser.add_argument('-mf', '--fid', nargs='*', default=['hfonly'],
                        help='Specify fidelity NN type [hfonly, seq-2net, seq-1net, hybrid-seq-2net, hybrid-dscr, dscr]')
    parser.add_argument('-lm', '--load_model_name', nargs='*', default=['comp.pt'],
                        help='Specify load model file name [Comp.pt] or the path of folders which contains model parameters [net_pars/]')
    parser.add_argument('-o', '--optimizer', nargs='*', default=['Adam'],
                        help='Specify the optimizer type [SGD or Adam or AdamW]')
    parser.add_argument('--test-input-xid', nargs='*', default=None,
                        help='Specify test input data xyzid if the data type is sqlite [xid_tst.txt]')
    parser.add_argument('--tr-input-xid', nargs='*', default=None,
                        help='Specify training(and validation) input data xyzid if the data type is sqlite [xid_tr.txt]')
    parser.add_argument('--trtsid-name', nargs='*', default=None,
                        help='Specify training(and validation)/test input data xyzid with tvt mask [trid_trtst.txt]')
    parser.add_argument('-if', '--fidlevel', default=None,
                        help='Specify input fidelity level (SQLite db only)')
    parser.add_argument('-nl', '--my-neurons', nargs='+', type=int, default=None,
                        help='Specify set of the number of neurons in each layer. The last number is for the output layer (new model only)')
    parser.add_argument('-al', '--my-actfn', nargs='+', default=None,
                        help='Specify set of the activation functions in each layer. [gaussian, tanh, identity, relu, silu, fsp] (new model only)')
    parser.add_argument('-sn', '--savenm', nargs='*', default='test',
                        help='Specify folder name to save the data. The optimizer and device will be appended to the name.')
    parser.add_argument('-r', '--randomseed', nargs='+', type=int, default=[0, 0], help='Specify random seed as integer')
    parser.add_argument('-nd', '--node', type=int, default=0, help='Specify gpu node as integer')
    parser.add_argument('-ls', '--lrscheduler', nargs='*', default=None,
                        help='Specify learning rate scheduler. [exp, step, rop]')
    parser.add_argument('-dr', '--decayrate', type=float, default=0.1,
                        help='Specify learning rate scheduler decay rate. default = 0.1')
    parser.add_argument('-ss', '--stepsize', type=int, default=5000,
                        help='Specify learning rate scheduler step size if using steplr. default = 5000')
    parser.add_argument('--tvt', nargs='+', type=float, default=[0.8, 0.1, 0.1],
                        help='Specify set of the decimal fraction (0.0,1.0] or number of points for training, validation and test set. The sum should be 1 if you input decimal points')
    parser.add_argument('--fw', type=float, default=0.5,
                        help='Specify the decimal fraction (0.0,1.0] for weighted loss of force. Only available when floss option is on')
    parser.add_argument('--floss', action='store_true', default=False, help='Use force in loss function')
    parser.add_argument('--wloss', action='store_true', default=False, help='Use weighted loss')
    parser.add_argument('-ns', '--nameset', nargs='+', default=None,
                        help='Specify set of the patterns of db name to generate redueced set. [w irc ...]')
    parser.add_argument('--savets', action='store_true', default=True, help='save test errors')
    parser.add_argument('--savevl', action='store_true', default=True, help='save validation errors')
    parser.add_argument('--savepr', action='store_true', default=False, help='save test predictions')
    parser.add_argument('--ftail', nargs='*', default=[''],
                        help='Specify any text to be added to the end of filename (flog, flog_ts, net_pars)')
    parser.add_argument('--stop-error', type=float, default=None,
                        help='Specify the value you want to stop the training when the model reaches a given error')
    parser.add_argument('--write-tvtmsk', action='store_true', default=False,
                        help='write training/validation/test mask')
    parser.add_argument('--read-trid', action='store_true', default=False,
                        help='read training set xyzid list')
    parser.add_argument('--stop-grad', action='store_true', default=False,
                        help='stop training if validation error increases. If you want to apply this training termination strategy after the training error reaches a given value, set \'stop-grad-err\'')
    parser.add_argument('--grad-trvl', action='store_true', default=False,
                        help='stop training if validation error increase is larger than training error decrease. If you want to apply this training termination strategy after the training error reaches a given value, set \'stop-grad-err\'')
    parser.add_argument('--stop-grad-err', type=float, default=None,
                        help='Specify the error criteria that you want to meet. Note that it is only valid with \'stop-grad\' option, and the training will not be terminated by validation error increase if the training error is larger than the value you set here.')
    parser.add_argument('--grad-step', type=int, default=1000,
                        help='Specify how often you want to check to stop the training using the error gradient if using stop-grad option. It should be greater than 200. default = 1000')
    parser.add_argument('--write_hdf', action='store_true', default=False,
                        help='save hdf5 file if sqlite was given as input data type')
    parser.add_argument('--temp', type=int, default=20000,
                        help='Specify temperature that QC data was sampled. default = 20000')
    parser.add_argument('-ofw', '--optimize-force-weight', action='store_true', default=False,              #Not yet sure how this works with multi fidelity.
                        help='optimize relative weights of energy and force in loss function')
    parser.add_argument('-sae', '--sae-fit', action='store_true', default=False,                            #Not yet sure how this works with multi fidelity.
                        help='compute single atom energies and subtract them from the dataset')
    parser.add_argument('-aev', '--aev_params', type=int, nargs='+',  default=[32, 8, 8],                            #Not yet sure how this works with multi fidelity.
                        help='parameters determining the length of the AEV')
    parser.add_argument('-R_c', '--cuttoff-radius', type=float, nargs='+',  default=[4.6, 3.1],                            #Not yet sure how this works with multi fidelity.
                        help='radial and angular cuttoff radii')
    parser.add_argument('--nbvlts', type=int, nargs='+',  default=[1, 1],                            #Use to avoid memory issues when doing force training.
                        help='number of batches to split validation and test set into')
    parser.add_argument('--no-biases', action='store_true', default=False,
                        help='dont include biases in NN')
    parser.add_argument('--delta', action='store_true', default=False,
                        help='delta learning to predict diference between 2 fidelity levels')
    parser.add_argument('-pb', '--pre-batched', action='store_true', default=False,
                        help='indicates that the force data has already been batched and saved in the proper location')
    
    
    parser.add_argument('-nh', '--my-neurons-hf', nargs='+', type=int, default=None,
                        help='Specify set of the number of neurons in each layer for high fidelity NN. The last number is for the output layer (new model only)')
    parser.add_argument('-ah', '--my-actfn-hf', nargs='+', default=None,
                        help='Specify set of the activation functions in each layer for high fidelity NN. [gaussian, tanh, identity, relu, silu, fsp] (new model only)')
    parser.add_argument('--tvt-lf', nargs='+', type=float, default=[0.8, 0.1, 0.1],
                        help='Specify set of the decimal fraction (0.0,1.0] or number of points for training, validation and test set for low fidelity data in multifidelity NN. The sum should be 1 if you input decimal points')
    parser.add_argument('--nested-lf', action='store_true', default=False, help='Sample low fidelity points that includes all high fidelity points')
    parser.add_argument('-dl', '--input-lf-data-type', choices=('xyz', 'aev', 'pca', 'sqlite'), default=None,
                        help='specify low fidelity input data file type [aev]')
    parser.add_argument('-fl', '--input-lf-data-fname', nargs='*', default=None,
                        help='Specify low fidelity input data filename(s) [aev_db.hdf5]')
    parser.add_argument('--trtsid-name-lf', nargs='*', default=None,
                        help='Specify training(and validation)/test input data xyzid with tvt mask for low fidelity [trid_trtst_lf.txt]')
    parser.add_argument('--fidlevel-lf', default=None,
                        help='Specify input low fidelity level (SQLite db only)')
    parser.add_argument('--refinehf', action='store_true', default=False,
                        help='Refine downstream network (high fidelity) only')
    parser.add_argument('--refinelf', action='store_true', default=False,
                        help='Refine upstream network (low fidelity) only')
    parser.add_argument('--costratio', type=float, default=1.,
                        help='Specify the cost ratio for weighted loss of multifidelity NN. default = 1.')
    parser.add_argument('--beta', type=float, default=0.95,
                        help='Scaling factor for angular component of SAE')

    args = parser.parse_args()
    print(args.input_data_fname)
    args.device = None
    if args.enable_cuda and torch.cuda.is_available():
        if args.node == 0:
            args.device = torch.device('cuda')
        else:
            args.device = torch.device('cuda:'+str(args.node))
    else:
        args.device = torch.device('cpu')

    if args.input_data_type == 'pca':
        args.input_data_fname[0] = 'aev_db_pca.hdf5'
    try:
        if args.lrscheduler[0] == 'None':
            args.lrscheduler = None
    except:
        pass
    return args


def args_to_actfn(argsact):
    actfn = []
    if isinstance(argsact, list):
        for i in range(0, len(argsact)):
            if argsact[i] == 'gaussian':
                actfn.append(nnpes.gaussian)
            elif argsact[i] == 'relu':
                actfn.append(nnpes.my_relu)
            elif argsact[i] == 'fsp':
                actfn.append(nnpes.fitted_softplus)
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

def model_to_archparams(model, args, hf=False):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = [np.prod(p.size()) for p in model_parameters]
    neurons = params[1::2]
    nhl = int(neurons.__len__() / 2)
    if args.fid[0] != 'hfonly':
        nhl_lf = int(neurons.__len__() / 4)
        if hf:
            return neurons[nhl_lf:nhl]
        else:
            return neurons[:nhl_lf]
    else:
        return neurons[:nhl]




def prep_netarch(args):
    if args.new_model:
        if args.my_neurons == None:
            args.my_neurons = [128, 64, 64, 1]
            args.my_activations = [nnpes.gaussian, nnpes.gaussian, nnpes.gaussian, nnpes.identity]
        else:
            if args.my_actfn == None:
                args.my_activations = args_to_actfn(len(args.my_neurons))
            else:
                args.my_activations = args_to_actfn(args.my_actfn)

    if args.new_model and args.fid[0] != 'hfonly':
        if args.my_neurons_hf == None:
            args.my_neurons_hf = args.my_neurons
            args.my_activations_hf = args.my_activations
            print('number of neurons and activaiton functions of hf NN are set to be same as that of lf NN.')
        else:
            if args.my_actfn_hf == None:
                args.my_activations_hf = args_to_actfn(len(args.my_neurons_hf))
            else:
                args.my_activations_hf = args_to_actfn(args.my_actfn_hf)
    if args.new_model and args.fid[0] == 'hfonly':
        args.my_neurons_hf = None
        args.my_activations_hf = None

    if args.new_model:
        print('number of neurons at each layers: ', args.my_neurons)
        print('activation functions:', args.my_activations)
        if args.fid[0] != 'hfonly':
            print('number of neurons at each layers in hf NN: ', args.my_neurons_hf)
            print('activation functions for hf NN:', args.my_activations_hf)

    if args.new_model:
        print('number of neurons at each layers: ', args.my_neurons)
        print('activation functions:', args.my_activations)
        if args.fid[0] != 'hfonly':
            print('number of neurons at each layers in hf NN: ', args.my_neurons_hf)
            print('activation functions for hf NN:', args.my_activations_hf)

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

def prep_tvtmsk_xyzid(args, dpes=None):
    print('prep_tvtmsk_xyzid: args.test_input_xid:',args.test_input_xid)
    if args.test_input_xid is not None:
        xid = read_xidtxt(args.test_input_xid[0])
        print('prep_tvtmsk_xyzid: len(xid):',len(xid))
    else:
        xid = None
        print('prep_tvtmsk_xyzid: xid is None')

    print('prep_tvtmsk_xyzid: args.tr_input_xid:',args.tr_input_xid)
    if args.tr_input_xid is not None:
        xid_tr = read_xidtxt(args.tr_input_xid[0])
        print('prep_tvtmsk_xyzid: len(xid_tr):',len(xid_tr))
    else:
        xid_tr = None
        print('prep_tvtmsk_xyzid: xid_tr is None')

    dbname = glob(args.input_data_fname[0])
    sname = args.savepth.split('/')[0] + '/tvt_tr{}.txt'
    print('prep_tvtmsk_xyzid: dbname:',dbname)
    print('prep_tvtmsk_xyzid:  sname:',sname)
    print('prep_tvtmsk_xyzid: calling data.write_tvtmsk_xyzid')
    sname_new = data.write_tvtmsk_xyzid(dpes, dbname, sname, fidlevel=args.fidlevel, testxid=xid, trxid=xid_tr, temp=args.temp, nameset=args.nameset)
    return sname_new


def sae_calculator(energies, atom_count, sae_guess=np.array([-1035.30436565, -16.8356588])): #sae_guess=np.array([-1028.6, -13.8])):

    def sae_fitting(sae_values, atom_count, total_energies):
            out_eng = np.zeros(len(total_energies))
            for i in range(len(sae_values)):
                out_eng += sae_values[i]*atom_count[:,i]
            return out_eng-total_energies

    #atom_count=np.array(atom_count)
    bounds=[]
    for i in range(len(sae_guess)):
        bounds.append(np.array([sae_guess[i]-3, sae_guess[i]+3]))
    bounds=tuple(bounds)
    print('SAE bounds: ', bounds)
    print('Performing least squares fitting to get SAE values to subtact from energies.')
    lsq_data = least_squares(sae_fitting, sae_guess, bounds=bounds, args=(atom_count, energies))
    #print('SAE energies: ', lsq_data.x / Hartree)
    print('SAE energies: ', lsq_data.x)
    return lsq_data.x


def prep_data(args, mf=None):

    # ==========================================================================================
    # instantiate data object for pes
    print("Instantiating PES data object")
    
    dpes = data.Data_pes()

    dpes.trid_fname = None
    args.excludexid = False
    # ============================================write_tvtmsk_xyzid==============================================
    # read database and parse it
    print("Get data")
    if args.wloss:
        # writes AEV with weights ('c' or 'z' for each class or each E zone):
        args.wrw = 'c'
    else:
        args.wrw = False
    if args.floss and not args.pre_batched:
        # write AEV with force:
        args.wrf = True
    else:
        args.wrf = False

    t00 = timeit.default_timer()
    if args.gen_byid:
        print('trainer: prep_data: calling read_xidtxt with args.trtsid_name[0]:',args.trtsid_name[0])
        xid = read_xidtxt(args.trtsid_name[0])
        print('trainer: prep_data: calling get_data with xid length:',len(xid))
        dpes.get_data(args, xid=xid)
    else:
        print('trainer: prep_data: calling get_data with no xid spec')
        dpes.get_data(args)

    t01 = timeit.default_timer()
    print('Generating AEV (min): ', (t01 - t00) / 60)

    
    # remove padding from dpes.full_symb_data that was needed to have variable sized molecules
    counter=0
    atom_count=[]
    for species in dpes.full_symb_data:
        dpes.full_symb_data[counter] = [element for element in species if element != '']
        atom_count.append([species.count('C'), species.count('H')])
        counter += 1

    

    try:
        args.alinit
        random.seed(0)
        dpes.random_shuffle_aev_db()
        new_ndat = sum(args.tvt)
        dpes.truncate_data(new_ndat)
    except:
        pass

    # do this for repeatability of random samples from pytorch
    print("For different train/test set distribution, setting random seed for pytorch, random, and numpy libraries")
    print('random seed:', args.randomseed[0])
    random.seed(args.randomseed[0])
    torch.manual_seed(random.randrange(200000))
    np.random.seed(random.randrange(200000))
    random.seed(random.randrange(200000))

    # randomly shuffle aev data base
    aev_db_shuffle = not args.dont_shuffle_data
    dpes.tvt_shuffle = not args.dont_shuffle_data
    if aev_db_shuffle:
        print("Shuffling aev db ...", end="")
        dpes.random_shuffle_aev_db()
        print("done")

    if args.truncate_data:
        if not aev_db_shuffle:
            print("WARNING: truncating data without prior random shuffling!!")
        print("ndat truncated from", dpes.ndat, end="")
        new_ndat = int(args.truncation_fraction * dpes.ndat)
        dpes.truncate_data(new_ndat)
        print(" to", dpes.ndat)

    ntvt = sum(args.tvt)
    if mf == 'lf':
        print('sum of tvt for low fidelity:', ntvt)
    elif mf == 'hf':
        print('sum of tvt for high fidelity:', ntvt)
    else:
        print('sum of tvt:', ntvt)


    if args.read_trid:
        if mf == 'lf':
            tridname = args.savepth.split('/')[0] + '/trid_lf_tr{}.txt'.format(str(int(ntvt)))
        elif mf == 'hf':
            tridname = args.savepth.split('/')[0] + '/trid_hf_tr{}.txt'.format(str(int(ntvt)))
        else:
            tridname = args.savepth.split('/')[0] + '/trid_tr{}.txt'.format(str(int(ntvt)))
        dpes.trid_fname = tridname
        print('tvtmsk file name: {}'.format(dpes.trid_fname))
    else:
        dpes.trid_fname = None

    if args.read_tvtmsk:
        tvtmsk = read_xidtxt(args.trtsid_name[0], tvtmsk=True)
        if mf == 'lf':
            if args.nested_lf:
                print('nested_lf was used. lf points are sampled based on hf training points ({}).'.format(args.trtsid_name[0]))
                ndat_hf = len([msk for msk in tvtmsk if msk[1] != '-1'])
                ndat_lf = int(sum(args.tvt))
                ndat_extra = ndat_lf - ndat_hf
                print('ndat from hf: {}\nndat newly sampled from lf: {}\n'.format(ndat_hf, ndat_extra), end='...')
                if ndat_extra > 0:
                    print('CALLING SAMPLE_XID with: ',dpes.meta, ndat_extra)
                    xid_extra = sample_xid.sample_xid(dpes.meta, ndat_extra, uniformdist=False)
                    tvt_extra = [['{}/{}'.format(*xx), '2'] for xx in xid_extra]
                print('{} of lf points were sampled..'.format(len(tvt_extra)))
                tvtmsk.extend(tvt_extra)
        dpes.set_tvt_mask(tvtmsk, tvt=args.tvt)
        print('trainer.py: prep_dat: tvt mask for testset was set based on the file: {}.'.format(args.trtsid_name))
    else:
        # define training, validation, and testing data subsets
        if ntvt == 1.:
            dpes.nvldat = int(args.tvt[1] * dpes.ndat)
            dpes.ntsdat = int(args.tvt[2] * dpes.ndat)
            dpes.ntrdat = dpes.ndat - dpes.ntsdat - dpes.nvldat
        else:
            dpes.ntrdat = int(args.tvt[0])
            dpes.nvldat = int(args.tvt[1])
            dpes.ntsdat = int(args.tvt[2])
            dpes.truncate_data(int(ntvt))
            print("ndat truncated to", dpes.ndat)
        dpes.set_tvt_mask()
    print("trainer.py: prep_dat: ntrdat:", dpes.ntrdat, "nvldat:", dpes.nvldat, "ntsdat:", dpes.ntsdat)

    # output actual data used (potentially shuffled and/or truncated)
    if not args.na:
        if mf == 'lf':
            actnm = "aev_db_lf_act.hdf5"
        elif mf == 'hf':
            actnm = "aev_db_hf_act.hdf5"
        else:
            actnm = "aev_db_act.hdf5"
        print("Packing data for writing {} -- actual shuffled/truncated data".format(actnm))
        dpes.pack_data(verbose=False)
        print("Writing {}".format(actnm))
        dpes.write_aev_db_hdf(actnm)
        print("done...")

    if savetrid:
        xyzid_E = [[int(d[-2].rsplit('/',1)[-1]), float(d[0])] for d in dpes.meta]
        id_tr = np.array(xyzid_E)
        np.savetxt('trid_all.txt', id_tr, fmt='%d %10.9f')

        xyzid_E_all = [[int(d[-2].rsplit('/',1)[-1]), float(d[0])] for d in dpes.meta]
        ids = [xyzid_E[i][0] for i in range(xyzid_E.__len__())]
        xyzid_ts = [a for a in xyzid_E_all if int(a[0]) not in ids]
        id_tr = np.array(xyzid_ts)
        np.savetxt('tsid_all.txt', id_tr, fmt='%d %10.9f')

    # define training batches
    print("trainer.py: prep_dat: Prepare training, validation, and testing data")
    if args.tr_batch_size:
        if mf == 'hf':
            num_tr_batches = args.num_tr_batches
        else:
            num_tr_batches = int(dpes.ntrdat / args.tr_batch_size)
            if dpes.ntrdat % args.tr_batch_size > 0:
                num_tr_batches += 1
    else:
        num_tr_batches = 1

    print("num_tr_batches:", num_tr_batches)
    dpes.prep_training_data(num_tr_batches, bpath = args.trpath)

    num_vl_batches = args.nbvlts[0]
    if dpes.nvldat > 0:
        print('trainer.py: prep_data: nvldat: ',dpes.nvldat)
        print('trainer.py: prep_data: calling dpes.prep_validation_data')
        dpes.prep_validation_data(num_vl_batches, bpath = args.vlpath)
        if args.savevl:
            print('Save validation results')

    num_ts_batches = args.nbvlts[1]
    if dpes.ntsdat > 0:
        dpes.prep_testing_data(num_ts_batches, bpath = args.tspath)
        if args.savets:
            print('Save test results')
    else:
        args.savets = False
    if args.write_tvtmsk:
        nn, mm = [dpes.full_symb_data[0].count(a) for a in dpes.atom_types]
        if mf == 'lf':
            args.tvtmsk_sname = args.savepth + 'tvtmsk_lf_tr' + str(int(dpes.ndat)) + '_rand' + str(
                args.randomseed[0]) + '.txt'
            data.write_tvtmsk_xyzid(dpes, str(Path.home()) + '/mlsdb/data/C5H5/C{}H{}.db'.format(str(nn), str(mm)),
                                    args.tvtmsk_sname, fidlevel=args.fidlevel_lf, temp=args.temp)
        elif mf == 'hf':
            args.tvtmsk_sname = args.savepth + 'tvtmsk_hf_tr' + str(int(dpes.ndat)) + '_rand' + str(
                args.randomseed[0]) + '.txt'
            data.write_tvtmsk_xyzid(dpes, str(Path.home()) + '/mlsdb/data/C5H5/C{}H{}.db'.format(str(nn), str(mm)),
                                    args.tvtmsk_sname, fidlevel=args.fidlevel, temp=args.temp)
        else:
            dbname = glob(args.input_data_fname[0])
            print('args.input_data_fname: ', args.input_data_fname)
            args.tvtmsk_sname = args.savepth + 'tvtmsk_tr' + str(int(dpes.ndat)) + '_rand' + str(args.randomseed[0]) + '.txt'
            data.write_tvtmsk_xyzid(dpes, dbname,
                                    args.tvtmsk_sname, fidlevel=args.fidlevel, temp=args.temp)
            #data.write_tvtmsk_xyzid(dpes, str(Path.home()) + '/mlsdb/data/C5H5/C{}H{}.db'.format(str(nn), str(mm)),
            #                        args.tvtmsk_sname, fidlevel=args.fidlevel, temp=args.temp)


    if args.wloss:
        if np.array(dpes.w).shape[0] < 0:
            print('weights are not available')
            sys.exit()
        else:
            print('weighted loss is used')
    if args.floss:
        #if dpes.fdat[0].__len__() > 0 and dpes.dxdat[0][0].__len__() > 0:
        print('force is included in loss function for training and weighted in E:F = {}:{}'.format(1.-args.fw, args.fw))
        dpes.xbtr = [[xt.requires_grad_() for xt in dpes.xbtr[b]] for b in range(dpes.nbttr)]
        if dpes.ntsdat > 0:
            dpes.xbts = [[xt.requires_grad_() for xt in dpes.xbts[b]] for b in range(dpes.nbtts)]
        if dpes.nvldat > 0:
            dpes.xbvl = [[xt.requires_grad_() for xt in dpes.xbvl[b]] for b in range(dpes.nbtvl)]
        #else:
        #    print('force data is not available')
        #    sys.exit()
    dpes.device = args.device

    dpes.indvout = False  # individual output

    if args.floss and not args.pre_batched:
        #dpes.fmax = np.max([np.max(dpes.fdat[i][0]) for i in range(dpes.ndat)])
        #dpes.fmin = np.min([np.min(dpes.fdat[i][0]) for i in range(dpes.ndat)])
        dpes.fmax = np.max([np.max(dpes.fdat[i][0]) for i in range(dpes.ndat)])
        dpes.fmin = np.min([np.min(dpes.fdat[i][0]) for i in range(dpes.ndat)])
        print('fmin: ',  dpes.fmin)
        print('fmax: ',  dpes.fmax)
        torch.save(torch.tensor([dpes.fmax, dpes.fmin]), args.savepth+'fmax_fmin.dat')
    elif args.pre_batched:
        fmax_fmin = torch.load(args.savepth+'fmax_fmin.dat')
        dpes.fmax = fmax_fmin[0]
        dpes.fmin = fmax_fmin[1]
        print('fmin: ',  dpes.fmin)
        print('fmax: ',  dpes.fmax)
    else:
        dpes.fmax = 1.
        dpes.fmin = 0.
    
    if not args.sae_fit:
        dpes.ymax = np.max([y[-1] for y in dpes.xdat])
        dpes.ymin = np.min([y[-1] for y in dpes.xdat])
   

    if args.pre_batched:
        dpes.pfdims_tr = torch.load(args.trpath+'pfims_tr')
        dpes.pfdims_vl = torch.load(args.vlpath+'pfims_vl')
        dpes.pfdims_ts = torch.load(args.tspath+'pfims_ts')
        #dpes.fbtr =  torch.load(args.trpath+'fbtr')
        #dpes.fbvl =  torch.load(args.trpath+'fbvl')
        #dpes.fbts =  torch.load(args.trpath+'fbts')
    

    dpes = var_be_gone(dpes)

    return dpes


def var_be_gone(dpes):
    from sys import getsizeof
    del dpes.tvtmsk
    del dpes.kkvl
    del dpes.kkts
    del dpes.kktr
    del dpes.meta
    del dpes.full_symb_data
    del dpes.pdat
    del dpes.xdat
    try:
        del dpes.fdat
        del dpes.padded_fdat
        del dpes.dxdat
        del dpes.padded_dxdat
        del dpes.full_symb_data_daev
        del dpes.fd2
    except:
        pass
    return dpes



# ====================================================================================================
def train_pes(args, verbose=True):
    print("In train_pes")
    # set default pytorch as double precision
    torch.set_default_dtype(torch.float64)

    # deal with cpu num threads spec
    maxthr = mp.cpu_count()  # On intel returns actually the max number of threads, not cpu-s
    if 2 * args.num_cores > maxthr:
        args.num_cores = int(maxthr / 2)
    torch.set_num_threads(
        args.num_cores)  # On intel uses 2x(argument) as number of threads, so actually give it num_cores

    # Set path
    args.savepth = args.savenm[0] + '_' + args.optimizer[0] + '_' + args.fid[0] + '_' + args.device.type + '/'
    args.trpath = args.savepth + 'training/'
    args.vlpath = args.savepth + 'validation/'
    args.tspath = args.savepth + 'testing/'
    Path(args.savepth).mkdir(parents=True, exist_ok=True)
    Path(args.trpath).mkdir(parents=True, exist_ok=True)
    Path(args.vlpath).mkdir(parents=True, exist_ok=True)
    Path(args.tspath).mkdir(parents=True, exist_ok=True)
    args.truncate_data = not (args.truncation_fraction == 1.)
    # Data preparation
    args.read_tvtmsk = False
    args.gen_byid = False
    if args.input_data_type == 'sqlite':
        if args.test_input_xid is not None:
            print("Running prep_tvtmsk_xyzid:")
            args.trtsid_name = [prep_tvtmsk_xyzid(args)]
            print('got trtsid_name:',args.trtsid_name)
            args.read_tvtmsk = True
            args.gen_byid = True
        elif args.trtsid_name is not None:
            args.read_tvtmsk = True
            args.gen_byid = True
    elif args.trtsid_name is not None:
        args.read_tvtmsk = True

    #if args.input_lf_data_fname != None:
    if args.input_lf_data_fname != None and not args.delta:
        ismultifidelity = True
        print('multifidelity input was parsed.')
    else:
        ismultifidelity = False

    if ismultifidelity or args.delta:
        print('is multifidelity')
        import copy
        args_lf = copy.copy(args)
        args_lf.input_data_fname = args.input_lf_data_fname
        args_lf.input_data_type = args.input_lf_data_type
        args_lf.tvt = args.tvt_lf
        args_lf.fidlevel = args.fidlevel_lf
        if args.nested_lf:
            args_lf.trtsid_name = args.trtsid_name
        else:
            if args.trtsid_name_lf == None:
                args_lf.read_tvtmsk = False
                args_lf.gen_byid = False
            args_lf.trtsid_name = args.trtsid_name_lf
        dpes_lf = prep_data(args_lf, mf='lf')

        if dpes_lf.ntsdat > 0:
            args.eval_err_lf = True
        else:
            args.eval_err_lf = False
        print('args.eval_err_lf: ', args.eval_err_lf)
        args.num_tr_batches = dpes_lf.nbttr
        dpes = prep_data(args, mf='hf')
        dpes.xbtr_lf = dpes_lf.xbtr
        dpes.indtr_lf = dpes_lf.indtr
        dpes.ybtr_lf = dpes_lf.ybtr
        dpes.ntrdat_lf = dpes_lf.ntrdat
        dpes.nbttr_lf = dpes_lf.nbttr
        dpes.bitr_lf = dpes_lf.bitr
        dpes.bftr_lf = dpes_lf.bftr
        dpes.fbtr_lf = dpes_lf.fbtr
        dpes.dxbtr_lf = dpes_lf.dxbtr
        dpes.fmax_lf = dpes_lf.fmax
        dpes.fmim_lf = dpes_lf.fmin
        #dpes.ymax_lf = dpes_lf.ymax
        #dpes.ymin_lf = dpes_lf.ymin
        #print(torch.sum(dpes.xbtr[0][0] - dpes.xbtr_lf[0][0]))
        
        #for i in range(len(dpes.xbtr_lf[0][0])):
        #    print(dpes.xbtr[0][0][i])
        #    print(dpes.xbtr_lf[0][0][i])
        #    print(torch.sum(dpes.xbtr[0][0][i] - dpes.xbtr_lf[0][0][i]))
        #for i in range(len(dpes.xbtr)):
        #    for j in range(len(dpes.xbtr[i])):
        #        if torch.equal(dpes.xbtr[i][j],dpes.xbtr_lf[i][j]):
        #            print('yep')
        #        else:
        #            print('nope')
        #sys.exit()
        #print('nbttr:', dpes.nbttr, '\nnbttr_lf:', dpes.nbttr_lf)
        try:
            dpes.xbts_lf = dpes_lf.xbts
            dpes.indts_lf = dpes_lf.indts
            dpes.ybts_lf = dpes_lf.ybts
            dpes.ntsdat_lf = dpes_lf.ntsdat
            dpes.nbtts_lf = dpes_lf.nbtts
            dpes.bits_lf = dpes_lf.bits
            dpes.bfts_lf = dpes_lf.bfts
            dpes.fbts_lf = dpes_lf.fbts
            dpes.dxbts_lf = dpes_lf.dxbts
        except:
            print('There is no test set in low fidelity data.')
        try:
            dpes.xbvl_lf = dpes_lf.xbvl
            dpes.indvl_lf = dpes_lf.indvl
            dpes.ybvl_lf = dpes_lf.ybvl
            dpes.nvldat_lf = dpes_lf.nvldat
            dpes.nbtvl_lf = dpes_lf.nbtvl
            dpes.bivl_lf = dpes_lf.bivl
            dpes.bfvl_lf = dpes_lf.bfvl
            dpes.fbvl_lf = dpes_lf.fbvl
            dpes.dxbvl_lf = dpes_lf.dxbvl
        except:
            print('There is no validation set in low fidelity data.')
    else:
        print('train_pes: calling single fidelity prep_data')
        dpes = prep_data(args, mf=None)
        args.eval_err_lf = False


    if ismultifidelity:
        print('ndat of high fidelity: {} \nndat of low fidelity: {}'.format(dpes.ntrdat, dpes.ntrdat_lf))

    if args.epochs < 1:
        print('Number of epochs is less than 1')
        print('To train a model the number of epochs should be > 0')
        sys.exit()

    # Prepare NN
    prep_netarch(args)

    load_model = not args.new_model

    random.seed(args.randomseed[1])
    torch.manual_seed(random.randrange(100000))
    np.random.seed(random.randrange(100000))
    random.seed(random.randrange(100000))


    if args.delta:
        all_trengs_shifted = []
        for b in range(len(dpes.ybtr)):
            for i in range(len(dpes.ybtr[b])):
                #print(dpes.ybtr[b][i].item(), dpes.ybtr_lf[b][i].item())
                dpes.ybtr[b][i]-=dpes.ybtr_lf[b][i]
                all_trengs_shifted.append(dpes.ybtr[b][i].item())
        for b in range(len(dpes.ybts)):
            for i in range(len(dpes.ybts[b])):
                #print(dpes.ybts[b][i].item(), dpes.ybts_lf[b][i].item())
                dpes.ybts[b][i]-=dpes.ybts_lf[b][i]
        for b in range(len(dpes.ybvl)):
            for i in range(len(dpes.ybvl[b])):
                #print(dpes.ybvl[b][i].item(), dpes.ybvl_lf[b][i].item())
                dpes.ybvl[b][i]-=dpes.ybvl_lf[b][i]
        if args.floss:
            for b in range(len(dpes.fbtr)):
                for i in range(len(dpes.fbtr[b])):
                    dpes.fbtr[b][i]-=dpes.fbtr_lf[b][i]
            
            for b in range(len(dpes.fbts)):
                for i in range(len(dpes.fbts[b])):
                    dpes.fbts[b][i]-=dpes.fbts_lf[b][i]

            for b in range(len(dpes.fbvl)):
                for i in range(len(dpes.fbvl[b])):
                    dpes.fbvl[b][i]-=dpes.fbvl_lf[b][i]
        all_trengs_shifted = np.array(all_trengs_shifted)
        dpes.ymax = np.max(all_trengs_shifted)
        dpes.ymin = np.min(all_trengs_shifted)
        print('dpes.ymin: ', dpes.ymin)
        print('dpes.ymax: ', dpes.ymax)
        print('Mean of subtracted energies (should be close to 0): ', np.mean(all_trengs_shifted))
        print('Range of subtracted energies: ', max(all_trengs_shifted) - min(all_trengs_shifted))



    #sae_late=True
    #if sae_late:
    if args.sae_fit:
        atom_count = []
        total_energies = []
        for b in range(len(dpes.ybtr)):
            Cvalues, Ccounts = np.unique(dpes.indtr[b][0], return_counts=True)
            Hvalues, Hcounts = np.unique(dpes.indtr[b][1], return_counts=True)
            for i in range(len(dpes.ybtr[b])):
                atom_count.append([Ccounts[i], Hcounts[i]])
                total_energies.append(dpes.ybtr[b][i][0])
        atom_count = np.array(atom_count)
        sae_energies = sae_calculator(total_energies, atom_count)
        energies_to_subtract = sae_energies[0] * atom_count[:,0] + sae_energies[1] * atom_count[:,1]
            
        subtracted_energies = total_energies - energies_to_subtract
            

        dpes.ymax = np.max(subtracted_energies)
        dpes.ymin = np.min(subtracted_energies)
        print('dpes.ymin: ', dpes.ymin)
        print('dpes.ymax: ', dpes.ymax)
        print('Mean of subtracted energies (should be close to 0): ', np.mean(subtracted_energies))
        print('Range of subtracted energies: ', max(subtracted_energies) - min(subtracted_energies))
        energy_counter = 0
        for b in range(len(dpes.ybtr)):
            for i in range(len(dpes.ybtr[b])):
                dpes.ybtr[b][i] = subtracted_energies[energy_counter]
                energy_counter += 1
        for b in range(len(dpes.ybvl)):
            Cvalues, Ccounts = np.unique(dpes.indvl[b][0], return_counts=True)
            Hvalues, Hcounts = np.unique(dpes.indvl[b][1], return_counts=True)
            for j in range(len(dpes.ybvl[b])):
                dpes.ybvl[b][j] = dpes.ybvl[b][j] - Ccounts[j] * sae_energies[0] - Hcounts[j] * sae_energies[1]
        for b in range(len(dpes.ybts)):
            Cvalues, Ccounts = np.unique(dpes.indts[b][0], return_counts=True)
            Hvalues, Hcounts = np.unique(dpes.indts[b][1], return_counts=True)
            for j in range(len(dpes.ybts[b])):
                dpes.ybts[b][j] = dpes.ybts[b][j] - Ccounts[j] * sae_energies[0] - Hcounts[j] * sae_energies[1]
        dpes.sae_energies = sae_energies

    else:
        dpes.sae_energies = np.zeros(dpes.num_nn)
    if args.sae_fit:
        if ismultifidelity:
            atom_count = []
            total_energies = []
            for b in range(len(dpes.ybtr_lf)):
                Cvalues, Ccounts = np.unique(dpes.indtr_lf[b][0], return_counts=True)
                Hvalues, Hcounts = np.unique(dpes.indtr_lf[b][1], return_counts=True)
                for i in range(len(dpes.ybtr_lf[b])):
                    atom_count.append([Ccounts[i], Hcounts[i]])
                    total_energies.append(dpes.ybtr_lf[b][i][0])
            atom_count = np.array(atom_count)
            sae_energies = sae_calculator(total_energies, atom_count)
            energies_to_subtract = sae_energies[0] * atom_count[:,0] + sae_energies[1] * atom_count[:,1]
            subtracted_energies = total_energies - energies_to_subtract


            dpes.ymax_lf = np.max(subtracted_energies)
            dpes.ymin_lf = np.min(subtracted_energies)
            print('dpes.ymin_lf: ', dpes.ymin_lf)
            print('dpes.ymax_lf: ', dpes.ymax_lf)
            print('Mean of subtracted energies (should be close to 0): ', np.mean(subtracted_energies))
            print('Range of subtracted energies: ', max(subtracted_energies) - min(subtracted_energies))
            energy_counter = 0
            for b in range(len(dpes.ybtr_lf)):
                for i in range(len(dpes.ybtr_lf[b])):
                    dpes.ybtr_lf[b][i] = subtracted_energies[energy_counter]
                    energy_counter += 1
            for b in range(len(dpes.ybvl)):
                Cvalues, Ccounts = np.unique(dpes.indvl_lf[b][0], return_counts=True)
                Hvalues, Hcounts = np.unique(dpes.indvl_lf[b][1], return_counts=True)
                for j in range(len(dpes.ybvl_lf[b])):
                    dpes.ybvl_lf[b][j] = dpes.ybvl_lf[b][j] - Ccounts[j] * sae_energies[0] - Hcounts[j] * sae_energies[1]
            for b in range(len(dpes.ybts)):
                Cvalues, Ccounts = np.unique(dpes_lf.indts[b][0], return_counts=True)
                Hvalues, Hcounts = np.unique(dpes_lf.indts[b][1], return_counts=True)
                for j in range(len(dpes.ybts_lf[b])):
                    dpes_lf.ybts[b][j] = dpes_lf.ybts[b][j] - Ccounts[j] * sae_energies[0] - Hcounts[j] * sae_energies[1]
            dpes_lf.sae_energies = sae_energies
    else:
        if ismultifidelity:
            total_energies = []
            for b in range(len(dpes.ybtr_lf)):
                for i in range(len(dpes.ybtr_lf[b])):
                    total_energies.append(dpes.ybtr_lf[b][i][0])
            dpes.ymax_lf = np.max(total_energies)
            dpes.ymin_lf = np.min(total_energies)
            dpes_lf.sae_energies = np.zeros(dpes.num_nn)
    
    

    include_biases = not args.no_biases
    if not load_model:
        nn_pes, criterion, optimizer = prep_model(load_model, args,
                                                  num_nn=dpes.num_nn, din_nn=dpes.dimdat,
                                                  my_neurons_a=args.my_neurons, my_activations_a=args.my_activations,
                                                  my_neurons_b=args.my_neurons, my_activations_b=args.my_activations,
                                                  my_neurons_hf_a=args.my_neurons_hf,
                                                  my_activations_hf_a=args.my_activations_hf,
                                                  my_neurons_hf_b=args.my_neurons_hf,
                                                  my_activations_hf_b=args.my_activations_hf,
                                                  sae_energies=dpes.sae_energies, biases=include_biases)
    else:
        nn_pes, criterion, optimizer = prep_model(load_model, args)
        args.my_neurons = model_to_archparams(nn_pes.net, args)
        args.my_activations = args_to_actfn(len(args.my_neurons))
        if ismultifidelity:
            args.my_neurons_hf = model_to_archparams(nn_pes.net, args, hf=True)
            args.my_activations_hf = args_to_actfn(len(args.my_neurons_hf))

    # Set savename
    if args.my_actfn == None:
        act = 'g'
    else:
        act = args.my_actfn[0][0]
    hns = act
    for i in range(len(args.my_neurons) - 1):
        hns = hns + str(args.my_neurons[i]) + '_'

    if ismultifidelity:
        if args.my_actfn_hf == None:
            act_hf = 'g'
        else:
            act_hf = args.my_actfn_hf[0][0]
        hns = 'lf_' + hns + 'hf_' + act_hf
        for i in range(len(args.my_neurons_hf) - 1):
            hns = hns + str(args.my_neurons_hf[i]) + '_'

    # if 'al' in args.savenm[0][:3]:
    #     args.ftail[0] = 'tr{}-{}'.format(dpes.ntrdat+dpes.nvldat, args.ftail[0])
    args.savepth_pars = 'net_pars_b' + str(args.tr_batch_size) + '_' + hns[:-1] + '_' + args.ftail[0] + '/'
    Path(args.savepth + args.savepth_pars).mkdir(parents=True, exist_ok=True)
    Path(args.savepth + args.savepth_pars + '/min/').mkdir(parents=True, exist_ok=True)
    if args.tvt[0] > 0:
        Path(args.savepth + args.savepth_pars + '/min_vl/').mkdir(parents=True, exist_ok=True)
    args.flognm = args.savepth + 'flog_b' + str(args.tr_batch_size) + '_' + hns[:-1] + '_' + args.ftail[0] + '.dat'
    if args.savets:
        args.flogtsnm = args.flognm.replace('flog', 'flog_ts')
    if args.savevl:
        args.flogvlnm = args.flognm.replace('flog', 'flog_vl')
    if args.savepr:
        args.flogprnm = args.flognm.replace('flog', 'flog_pr')

    if verbose:
        print("device:", args.device)
        print("Machine core_count:", int(maxthr / 2), "i.e.", maxthr, "threads")
        if 2 * args.num_cores > maxthr:
            print("Requested number of cores (", args.num_cores, ") unavailable.")
        print("Using", args.num_cores, "cores, i.e.", 2 * args.num_cores, "threads")

        print("input_data_type:", args.input_data_type)
        print("input_data_fname:", args.input_data_fname)
        if args.input_lf_data_fname != None:
            print('low fidelity data file name and type: ', args.input_lf_data_fname, ', ', args.input_lf_data_type)
        else:
            print('low fidelity data filename is not specified.')

        print("truncate_data:", args.truncate_data)
        if args.truncate_data:
            print("truncation_fraction:", args.truncation_fraction)

        print("aev_db_shuffle:", not args.dont_shuffle_data)

        if args.nw:
            print("Not writing net & opt files")
        else:
            print("Saving NNs and optimizer")
            nn_pes.save(args, args.savepth + args.savepth_pars + "comp", 0, optimizer)
            nn_pes.print_model(args.savepth + args.savepth_pars + "comp", 0)
            print_opt(optimizer, args.savepth + args.savepth_pars + "optm", 0)

        if args.no_output_pred:
            print("Not calling output_pred")

        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            wd = param_group['weight_decay']
        print("learning_rate:", lr)

        print("momentum:", args.momentum)
        print('L2 penalty:', wd)
        print("epochs:", args.epochs)
        print("ntlp:", args.ntlp)
        print("save-every (model):", args.save_every)
        print("Batch size for training (roughly):", args.tr_batch_size)

        print("ni=", args.ni)
        print("nf=", args.nf)

        print("load_model:", load_model, '--', args.load_model_name)

        print('optimizer is {}.'.format(args.optimizer[0]))

    t0 = timeit.default_timer()
    if 'cuda' in args.device.type:
        nn_pes.net.to(args.device)
        print('NN was moved to cuda:', next(nn_pes.net.parameters()).is_cuda)
        #print('dpes.__dict__: ', dpes.__dict__)
        dpes.tocuda(wloss=args.wloss, floss=args.floss)

    t1 = timeit.default_timer()
    print('Time to transfer the tensors from cpu to gpu: ', t1 - t0)

    tic = time.perf_counter()
    print("Training NNs, epochs:", args.epochs, ", batches:", dpes.nbttr, ", ntrdat:", dpes.ntrdat)
    train(nn_pes, dpes, criterion, optimizer, args)

    toc = time.perf_counter()
    time_min = (toc - tic) / 60


    if (dpes.nvldat > 0):
        print("Validating NNs")
        validate(nn_pes, dpes, criterion)

    if (dpes.ntsdat > 0):
        print("Testing NNs")
        test(nn_pes, dpes, criterion, outlf=args.eval_err_lf)

    if not args.no_output_pred:
        output_pred(nn_pes, dpes, outlf=args.eval_err_lf)

    # print('E_tr: {} \nE_ts: {}'.format(dpes.ybtr, dpes.ybts))

    print('{} minute was taken for the calculation.'.format(time_min))

    return 1
# ====================================================================================================
def main():
    # set default pytorch as double precision
    torch.set_default_dtype(torch.float64)

    # preamble -- handling arguments
    args = parse_arguments_list()
    train_pes(args)

if __name__ == "__main__":
    main()
