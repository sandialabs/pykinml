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
import itertools
import random
import pickle
import timeit
import time

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp


from pykinml import data
from pykinml import nnpes
from pykinml import daev as daev_calc
from pykinml import prepper as prep
from pykinml.prepper import kcpm

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



# ====================================================================================================


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12334"
    init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)




# ====================================================================================================

# ====================================================================================================


class Runner:
    def __init__(self, args, device):
        self.num_spec = len(args.present_elements)
        self.model = prep.load_trained_model(args, load_opt=False)
        self.model.to(device)
        self.model.add_sae = True
 
        self.device = device
        self.args = args


    def get_ids(self, aevs):
        ids=[[] for i in range(self.num_spec)]
        for mol in range(len(aevs)):
            for spec in range(self.num_spec):
                ids[spec] += [mol] * len(aevs[mol][spec])
        ids = [torch.tensor(ida).to(self.device) for ida in ids]
        return [ids]

    def eval_force(self, daevs):
        self.pred_forces = []
        for i in range(len(daevs)):
            force = daev_calc.cal_dEdxyz_ddp(self.aevs[i], -self.pred_engs[i], daevs[i], self.ids[i])
            self.pred_forces.append(force)
        return self.pred_forces

    def eval(self, aevs, daevs=[]):
        self.ids = self.get_ids(aevs)
        self.pred_engs = []
        self.pred_engs_lf = []
        self.aevs = aevs
        for i in range(len(aevs)):
            if self.args.multi_fid:
                eng_lf, eng = self.model(self.aevs[i], self.ids[i], 1, train=False)
                self.pred_engs_lf.append(eng_lf)
            else:
                eng = self.model(self.aevs[i], self.ids[i], 1, train=False)
            self.pred_engs.append(eng)
        return torch.cat(self.pred_engs)



class Trainer:
    def __init__(self, model, train_data, valid_data, test_data, keys, optimizer, lr_scheduler, sae_energies, save_every: int, fname: str, svpath:str, device, force_train=False, num_spec=2):
        """
        Class for training single fidelity NNPES.
        input:
            model: instance of nnpes.CompositeNetworks class. Can be wrapped in pytorch DDP.
            train_data: instance of pytorch DataLoader. AEVs and energies of training data. If force_train=True must also include forces, aev derivatives, and length of true forces (fdims))
            vald_data: instance of pytorch DataLoader. AEVs and energies of validation data. If force_train=True must also include forces, aev derivatives, and length of true forces (fdims))
            test_data: instance of pytorch DataLoader. AEVs and energies of test data. If force_train=True must also include forces, aev derivatives, and length of true forces (fdims))

        """
        
        random.seed(0)
        torch.manual_seed(random.randrange(200000))
        np.random.seed(random.randrange(200000))
        random.seed(random.randrange(200000))
        self.device = device
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.num_spec = num_spec
        self.keys = keys
        self.save_every = save_every
        self.FT=force_train
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.sae_energies = sae_energies
        self.fname = fname
        self.svpath = svpath# + '_' + 'device' + '_' + str(self.device) + '_' + str(rank)
        if self.device == 'cpu' or self.device == 0:
            Path(self.svpath).mkdir(parents=True, exist_ok=True)
        self.openf = open(fname, "w")


    def save_ddp(self, net_fnam, epoch, optimizer):
        torch.save({'epoch': epoch,
                    'optimizer': optimizer,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler,
                    'model_state_dict': self.model.state_dict(),
                    'sae_energies': self.sae_energies,
                    'params': self.model.module.netparams
                    },
                   self.svpath + '/' + net_fnam + '-' + str(epoch).zfill(4) + ".pt")

    def save_cpu(self, net_fnam, epoch, optimizer):
        torch.save({'epoch': epoch,
                    'optimizer': optimizer,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler,
                    'model_state_dict': self.model.state_dict(),
                    'sae_energies': self.sae_energies,
                    'params': self.model.netparams
                    },
                   self.svpath + '/' + net_fnam + '-' + str(epoch).zfill(4) + ".pt")


    def cat_data(self, tocat, key, ind):
        random.seed(0)
        torch.manual_seed(random.randrange(200000))
        np.random.seed(random.randrange(200000))
        random.seed(random.randrange(200000))
        items = []
        ids = []
        nmols = []
        for batch_all in tocat:
            batch = [batch_all[i][ind] for i in range(len(batch_all))]
            if key == 'engs' or key == 'forces' or key == 'daevs':
                batch = [torch.tensor(i) for i in batch]
            if key == 'aevs' or key == 'daevs':
                ss = [[[] for mol in range(len(batch))] for spec in range(self.num_spec)]
                id_spec = [[] for spec in range(self.num_spec)]
                mol_count = 0
                for mol in range(len(batch)):
                    ss_spec = []
                    mol_count += 1
                    for spec in range(len(batch[mol])):
                        ss[spec][mol] = batch[mol][spec].to(self.device).requires_grad_()
                        if key == 'aevs':
                            id_spec[spec] += [mol] * len(batch[mol][spec])
                ss = [torch.cat(s) for s in ss]
                bitem = ss
                if key == 'aevs':
                    id_spec = [torch.tensor(spec).to(self.device) for spec in id_spec]
                    ids.append(id_spec)
                    nmols.append(mol_count)
            
            if key == 'forces' or key=='engs':
                bitem = torch.cat([item for item in batch]).to(self.device)
                if key == 'engs': 
                    bitem = bitem.unsqueeze(0).T
            if key == 'fdims':
                bitem = batch
            items.append(bitem)
        if key == 'aevs':
            return items, ids, nmols
        else:
            return items


    def _run_batch_tvs(self, tvs, aevs, ids, nmols, true_engs, true_forces=[], fdims=[], daevs=[]):

        if tvs == 'train':
            self.optimizer.zero_grad()
        pred_engs, log_sigma = self.model(aevs, ids, nmols)
        ediff = prep.energy_abs_dif(pred_engs, true_engs)
        fdiff=[]        #This is a placeholder for when not doing force training
        if self.FT:
            pred_forces = daev_calc.cal_dEdxyz_ddp(aevs, -pred_engs, daevs, ids)
            fdiff = prep.force_abs_dif(pred_forces, true_forces, fdims)
        if tvs == 'train':
            eloss, floss = prep.my_loss(ediff, fdiff, dEsq=1., dfsq=1., p=2)
            loss = self.model.mtl([eloss, floss], log_sigma)
            self.log_sigma = log_sigma
            loss.backward(retain_graph=True)
            loss.retain_grad()
            self.optimizer.step()
        return ediff, fdiff
    


    def _run_epoch(self, epoch, data, ids, nmols, tvs):
        bat_lst = list(range(len(data['engs'])))
        ediff = []
        fdiff = []
        for b in bat_lst:
            if self.FT:
                bediff, bfdiff = self._run_batch_tvs(tvs, data['aevs'][b], ids[b], nmols[b], data['engs'][b], data['forces'][b], data['fdims'][b], data['daevs'][b])
                fdiff += bfdiff
            else:
                bediff, bfdiff = self._run_batch_tvs(tvs, data['aevs'][b], ids[b], nmols[b], data['engs'][b])
            ediff += bediff
        ediff = torch.tensor(ediff)
        L1 = torch.mean(ediff)
        L2 = torch.sqrt(torch.mean(ediff**2))
        Linf = torch.max(ediff)
        self.openf.write(tvs + ' L1 loss: ' + str(kcpm(L1.item())) + '\n')
        self.openf.write(tvs + ' L2 loss: ' + str(kcpm(L2.item())) + '\n')
        self.openf.write(tvs + ' Linf loss: ' + str(kcpm(Linf.item())) + '\n')
        print('device ', self.device, ' ',tvs + ' MAE: ', kcpm(L1.item()))
        print('device ', self.device, ' ',tvs + ' RMSE: ', kcpm(L2.item()))
        print('device ', self.device, ' ',tvs + ' Linf: ', kcpm(Linf.item()))
        if self.FT:
            fdiff = torch.cat(fdiff)
            fL1 = torch.mean(fdiff)
            fL2 = torch.sqrt(torch.mean(fdiff**2))
            fLinf = torch.max(fdiff)
            self.openf.write(tvs + ' fL1 loss: ' + str(kcpm(fL1.item())) + '\n')
            self.openf.write(tvs + ' fL2 loss: ' + str(kcpm(fL2.item())) + '\n')
            self.openf.write(tvs + ' fLinf loss: ' + str(kcpm(fLinf.item())) + '\n')
            print('device ', self.device, ' ',tvs + ' FMAE: ', kcpm(fL1.item()))
            print('device ', self.device, ' ',tvs + ' FRMSE: ', kcpm(fL2.item()))
            print('device ', self.device, ' ',tvs + ' FLinf: ', kcpm(fLinf.item()))
            self.openf.write('log_sigma: ')
            for task in range(len(self.log_sigma)):
                self.openf.write(str(self.log_sigma[task].item())+' ')
        self.openf.write('\n')
        if tvs == 'valid':
            self.lr_scheduler.step(kcpm(L2))



    def train(self, max_epochs: int):
        tr0 = time.time()
        random.seed(0)
        torch.manual_seed(random.randrange(200000))
        np.random.seed(random.randrange(200000))
        random.seed(random.randrange(200000))
        self.openf.write('Initial parameters for device: '+ str(self.device) + '\n')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.openf.write(name + ': ' + str(param.data) + '\n')
        
        print('self.keys: ', self.keys)
        train_dict = {} 
        valid_dict = {}
        test_dict = {}

        
        for i in range(len(self.keys)):
            if self.keys[i] == 'aevs':
                train_dict[self.keys[i]], tr_ids, tr_nmols = prep.cat_data(self.train_data, self.keys[i], i, self.device, self.num_spec)#self.cat_data(self.train_data, self.keys[i], i)
                valid_dict[self.keys[i]], vl_ids, vl_nmols = prep.cat_data(self.valid_data, self.keys[i], i, self.device, self.num_spec)#self.cat_data(self.valid_data, self.keys[i], i)
                test_dict[self.keys[i]], ts_ids, ts_nmols = prep.cat_data(self.test_data, self.keys[i], i, self.device, self.num_spec)#self.cat_data(self.test_data, self.keys[i], i)
            else:
                train_dict[self.keys[i]] = prep.cat_data(self.train_data, self.keys[i], i, self.device, self.num_spec)#self.cat_data(self.train_data, self.keys[i], i)
                valid_dict[self.keys[i]] = prep.cat_data(self.valid_data, self.keys[i], i, self.device, self.num_spec)#self.cat_data(self.valid_data, self.keys[i], i)
                test_dict[self.keys[i]] = prep.cat_data(self.test_data, self.keys[i], i, self.device, self.num_spec)#self.cat_data(self.test_data, self.keys[i], i)



        for epoch in range(max_epochs):
            print('\ndevice ', self.device,' epoch: ', epoch)
            self.openf.write('\nepoch: ' + str(epoch) + '\n')
            self._run_epoch(epoch, train_dict, tr_ids, tr_nmols, tvs='train')
            self._run_epoch(epoch, valid_dict, vl_ids, vl_nmols, tvs='valid')
            if self.device == 0 or self.device == 'cpu':
                self._run_epoch(epoch, test_dict, ts_ids, ts_nmols, tvs='test')
            
            if self.device == 'cpu' and (epoch % self.save_every == 0 or epoch  + 1 == max_epochs):
                self.save_cpu('model', epoch, self.optimizer)
            if self.device == 0 and (epoch % self.save_every == 0 or epoch  + 1 == max_epochs): 
                self.save_ddp('model', epoch, self.optimizer)

        print('device Number ', self.device, 'is DONE!!!')
        self.openf.write('Final parameters for gpu: ' +  str(self.device) + '\n')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.openf.write(name + ': ' + str(param.data) + '\n')
        
        tr1 = time.time()
        self.openf.write('Training time: ' + str(tr1-tr0))
        self.openf.close()



# ====================================================================================================
def spawned_trainer(rank, args, world_size:int):
    """
    This function loads the initializes the training loop. It calls the functions nneeded to load the preprepared
    data and build the model.
    Inputs:
        rank:
            The device used for training.
            if args.ddp=True (int): the index of the gpu (handled by torch.multiprocessing in example.py)
            else (str): cpu

        args:
            instance of prepper.parse_arguments_list() holding arguments needed for training.

        world_size (int):
            Number of GPUs two be used.
            if args.ddp=True: match torch.cuda.device_count(). In train_single_fid.py, os.environ['CUDA_VISIBLE_DEVICES'] is set using args.gpus to limit the number of devices to only those requested.
            else: 0

    """
    # set default pytorch as double precision
    print('rank: ', rank)
    random.seed(args.randomseed[1])
    torch.manual_seed(random.randrange(200000))
    np.random.seed(random.randrange(200000))
    random.seed(random.randrange(200000))
    torch.set_default_dtype(torch.float64)
    #torch.set_printoptions(precision=8)
    seeds = args.randomseed
    args.num_species = len(args.present_elements)
    print('torch.cuda.is_available(): ', torch.cuda.is_available())
    if args.ddp:
        print('setting up DDP!')
        ddp_setup(rank, world_size)
        print('DDP has been setup!')
    fname = args.savenm + '_device_' + str(rank) + '.log'
    print('about to load train objects!')
    save_path = args.savenm# + '_seeds_'+str(seeds[0])+'_'+str(seeds[1])
    torch.autograd.set_detect_anomaly(True)
    train_set, valid_set, test_set, model, optimizer, lr_scheduler, sae_energies, keys = prep.load_train_objs(args, fid=args.fidlevel)
    prep.set_up_task_weights(model, args, optimizer)
    device = rank
    if args.ddp:
        model = DDP(model.to(device), device_ids=[device], find_unused_parameters=True)
        model.mtl = model.module.mtl
    else:
        model = model.to(device)
    trainer = Trainer(model, train_set, valid_set, test_set, keys, optimizer, lr_scheduler, sae_energies, args.save_every, fname, save_path, device, force_train=args.floss, num_spec = args.num_species) 
    trainer.train(args.epochs)
    if args.ddp:
        destroy_process_group()

def main(args):
    tt1 = time.time()
    torch.set_default_dtype(torch.float64)
    if args.ddp:
        args.gpus = [str(g) for g in args.gpus]
        args.gpus = ', '.join(args.gpus)
        print('GPUS: ', args.gpus)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if not args.pre_saved:
        print('preping data')
        prep.prep_data(args)
        args.pre_saved = True
    world_size = torch.cuda.device_count()
    print('world_size: ', world_size)
    if args.ddp:
        mp.spawn(spawned_trainer, args=[args,world_size], nprocs=world_size, join=True)
    else:
        spawned_trainer('cpu', args, 0)
    tt2 = time.time()
    print('Time it took: ', tt2-tt1)

if __name__ == "__main__":
    # set default pytorch as double precision
    torch.set_default_dtype(torch.float64)

    # preamble -- handling arguments
    args = prep.parse_arguments_list()
    main(args)



