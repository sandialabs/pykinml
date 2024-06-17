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
from ase.units import mol, kcal

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



class Trainer_mf:
    def __init__(self, model, train_data, valid_data, test_data, keys, train_data_lf, valid_data_lf, test_data_lf, keys_lf, optimizer, lr_scheduler, sae_energies, save_every: int, fname: str, svpath:str, device, force_train=False, num_spec=2):
        random.seed(0)
        torch.manual_seed(random.randrange(200000))
        np.random.seed(random.randrange(200000))
        random.seed(random.randrange(200000))
        self.device = device
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.keys = keys

        self.train_data_lf = train_data_lf
        self.valid_data_lf = valid_data_lf
        self.test_data_lf = test_data_lf
        self.keys_lf = keys_lf

        self.num_spec = num_spec
        self.save_every = save_every
        self.FT=force_train
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.sae_energies = sae_energies
        self.fname = fname
        self.svpath = svpath
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
            batch = np.array(batch_all)[:,ind]
            if key == 'aevs' or key == 'daevs':
                ss = [[[] for mol in range(len(batch))] for spec in range(self.num_spec)]
                id_spec = [[] for spec in range(self.num_spec)]
                mol_count = 0
                for mol in range(len(batch)):
                    mol_count += 1
                    for spec in range(len(batch[mol])):
                        if key == 'daevs':
                            ss[spec][mol] = batch[mol][spec].to(self.device)
                        if key == 'aevs':
                            ss[spec][mol] = batch[mol][spec].to(self.device).requires_grad_()
                            id_spec[spec] += [mol] * len(batch[mol][spec])
                ss = [torch.cat(s) for s in ss]
                bitem = ss
                if key == 'aevs':
                    id_spec = [torch.tensor(spec).to(self.device) for spec in id_spec]
                    ids.append(id_spec)
                    nmols.append(mol_count)
            
            if 'forces' in key or 'engs' in key:
                bitem = torch.cat([item for item in batch]).to(self.device)
                if 'engs' in key:
                    bitem = bitem.unsqueeze(0).T
            if 'fdims' in key:
                bitem = batch
            items.append(bitem)
        if key == 'aevs':
            return items, ids, nmols
        else:
            return items


    def _run_batch_tvs(self, tvs, aevs, ids, nmols, true_engs_lf, true_engs_hf, true_forces_lf=[], true_forces_hf=[], fdims=[], daevs=[]):

        if tvs == 'train':
            self.optimizer.zero_grad()
        pred_engs_lf, pred_engs_hf, log_sigma = self.model(aevs, ids, nmols)
        ediff_lf = prep.energy_abs_dif(pred_engs_lf, true_engs_lf)
        ediff_hf = prep.energy_abs_dif(pred_engs_hf, true_engs_hf)
        fdiff_lf=[]        #This is a placeholder for when not doing force training
        fdiff_hf=[]        #This is a placeholder for when not doing force training
        if self.FT:
            pred_forces_lf = daev_calc.cal_dEdxyz_ddp(aevs, -pred_engs_lf, daevs, ids)
            fdiff_lf = prep.force_abs_dif(pred_forces_lf, true_forces_lf, fdims)
            pred_forces_hf = daev_calc.cal_dEdxyz_ddp(aevs, -pred_engs_hf, daevs, ids)
            fdiff_hf = prep.force_abs_dif(pred_forces_hf, true_forces_hf, fdims)
        
        if tvs == 'train':
            eloss_lf, floss_lf = prep.my_loss(ediff_lf, fdiff_lf, dEsq=1., dfsq=1., p=2)
            eloss_hf, floss_hf = prep.my_loss(ediff_hf, fdiff_hf, dEsq=1., dfsq=1., p=2)
            eloss = eloss_lf + eloss_hf
            floss = floss_lf + floss_hf
            loss = self.model.mtl([eloss, floss], log_sigma)
            self.log_sigma = log_sigma
            loss.backward(retain_graph=True)
            loss.retain_grad()
            self.optimizer.step()
        return ediff_lf, fdiff_lf, ediff_hf, fdiff_hf
    


    def _run_epoch(self, epoch, data, ids, nmols, tvs, data_lf={}):
        bat_lst = list(range(len(data['engs'])))
        ediff_lf = []
        fdiff_lf = []
        ediff_hf = []
        fdiff_hf = []
        for b in bat_lst:
            if self.FT:
                bediff_lf, bfdiff_lf, bediff_hf, bfdiff_hf = self._run_batch_tvs(tvs, data['aevs'][b], ids[b], nmols[b], data_lf['engs'][b], data['engs'][b], data_lf['forces'][b], data['forces'][b], data['fdims'][b], data['daevs'][b])
                fdiff_lf += bfdiff_lf
                fdiff_hf += bfdiff_hf
            else:
                bediff_lf, bfdiff_lf, bediff_hf, bfdiff_hf = self._run_batch_tvs(tvs, data['aevs'][b], ids[b], nmols[b], data['engs'][b], data_lf['engs'][b])
            ediff_lf += bediff_lf
            ediff_hf += bediff_hf
        ediff_lf = torch.tensor(ediff_lf)
        L1_lf = torch.mean(ediff_lf)
        L2_lf = torch.sqrt(torch.mean(ediff_lf**2))
        Linf_lf = torch.max(ediff_lf)
        self.openf.write(tvs + ' L1_lf loss: ' + str(kcpm(L1_lf.item())) + '\n')
        self.openf.write(tvs + ' L2_lf loss: ' + str(kcpm(L2_lf.item())) + '\n')
        self.openf.write(tvs + ' Linf_lf loss: ' + str(kcpm(Linf_lf.item())) + '\n')
        print('device ', self.device, ' ',tvs + ' MAE_lf: ', kcpm(L1_lf.item()))
        print('device ', self.device, ' ',tvs + ' RMSE_lf: ', kcpm(L2_lf.item()))
        print('device ', self.device, ' ',tvs + ' Linf_lf: ', kcpm(Linf_lf.item()))

        ediff_hf = torch.tensor(ediff_hf)
        L1_hf = torch.mean(ediff_hf)
        L2_hf = torch.sqrt(torch.mean(ediff_hf**2))
        Linf_hf = torch.max(ediff_hf)
        self.openf.write(tvs + ' L1_hf loss: ' + str(kcpm(L1_hf.item())) + '\n')
        self.openf.write(tvs + ' L2_hf loss: ' + str(kcpm(L2_hf.item())) + '\n')
        self.openf.write(tvs + ' Linf_hf loss: ' + str(kcpm(Linf_hf.item())) + '\n')
        print('device ', self.device, ' ',tvs + ' MAE_hf: ', kcpm(L1_hf.item()))
        print('device ', self.device, ' ',tvs + ' RMSE_hf: ', kcpm(L2_hf.item()))
        print('device ', self.device, ' ',tvs + ' Linf_hf: ', kcpm(Linf_hf.item()))

        if self.FT:
            fdiff_lf = torch.cat(fdiff_lf)
            fL1_lf = torch.mean(fdiff_lf)
            fL2_lf = torch.sqrt(torch.mean(fdiff_lf**2))
            fLinf_lf = torch.max(fdiff_lf)
            self.openf.write(tvs + ' fL1_lf loss: ' + str(kcpm(fL1_lf.item())) + '\n')
            self.openf.write(tvs + ' fL2_lf loss: ' + str(kcpm(fL2_lf.item())) + '\n')
            self.openf.write(tvs + ' fLinf_lf loss: ' + str(kcpm(fLinf_lf.item())) + '\n')
            print('device ', self.device, ' ',tvs + ' FMAE_lf: ', kcpm(fL1_lf.item()))
            print('device ', self.device, ' ',tvs + ' FRMSE_lf: ', kcpm(fL2_lf.item()))
            print('device ', self.device, ' ',tvs + ' FLinf_lf: ', kcpm(fLinf_lf.item()))

            fdiff_hf = torch.cat(fdiff_hf)
            fL1_hf = torch.mean(fdiff_hf)
            fL2_hf = torch.sqrt(torch.mean(fdiff_hf**2))
            fLinf_hf = torch.max(fdiff_hf)
            self.openf.write(tvs + ' fL1_hf loss: ' + str(kcpm(fL1_hf.item())) + '\n')
            self.openf.write(tvs + ' fL2_hf loss: ' + str(kcpm(fL2_hf.item())) + '\n')
            self.openf.write(tvs + ' fLinf_hf loss: ' + str(kcpm(fLinf_hf.item())) + '\n')
            print('device ', self.device, ' ',tvs + ' FMAE_hf: ', kcpm(fL1_hf.item()))
            print('device ', self.device, ' ',tvs + ' FRMSE_hf: ', kcpm(fL2_hf.item()))
            print('device ', self.device, ' ',tvs + ' FLinf_hf: ', kcpm(fLinf_hf.item()))

            self.openf.write('log_sigma: ')
            for task in range(len(self.log_sigma)):
                self.openf.write(str(self.log_sigma[task].item())+' ')
        self.openf.write('\n')
        if tvs == 'valid':
            self.lr_scheduler.step(kcpm(L2_hf))



    def train(self, max_epochs: int):
        tr0 = time.time()
        random.seed(0)
        torch.manual_seed(random.randrange(200000))
        np.random.seed(random.randrange(200000))
        random.seed(random.randrange(200000))
        self.openf.write('Initial parameters for gpu: '+ str(self.device) + '\n')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.openf.write(name + ': ' + str(param.data) + '\n')
        
        train_dict = {}
        valid_dict = {}
        test_dict = {}
        print('self.keys: ', self.keys)
        print('self.keys_lf: ', self.keys_lf)
        

        for i in range(len(self.keys)):
            if self.keys[i] == 'aevs':
                train_dict[self.keys[i]], tr_ids, tr_nmols = prep.cat_data(self.train_data, self.keys[i], i, self.device, self.num_spec)#self.cat_data(self.train_data, self.keys[i], i)
                valid_dict[self.keys[i]], vl_ids, vl_nmols = prep.cat_data(self.valid_data, self.keys[i], i, self.device, self.num_spec)#self.cat_data(self.valid_data, self.keys[i], i)
                test_dict[self.keys[i]], ts_ids, ts_nmols = prep.cat_data(self.test_data, self.keys[i], i, self.device, self.num_spec)#self.cat_data(self.test_data, self.keys[i], i)
            else:
                train_dict[self.keys[i]] = prep.cat_data(self.train_data, self.keys[i], i, self.device, self.num_spec)#self.cat_data(self.train_data, self.keys[i], i)
                valid_dict[self.keys[i]] = prep.cat_data(self.valid_data, self.keys[i], i, self.device, self.num_spec)#self.cat_data(self.valid_data, self.keys[i], i)
                test_dict[self.keys[i]] = prep.cat_data(self.test_data, self.keys[i], i, self.device, self.num_spec)#self.cat_data(self.test_data, self.keys[i], i)


        train_dict_lf = {}
        valid_dict_lf = {}
        test_dict_lf = {}

        for i in range(len(self.keys_lf)):
            if self.keys_lf[i] == 'aevs':
                train_dict_lf[self.keys_lf[i]], tr_ids, tr_nmols = prep.cat_data(self.train_data_lf, self.keys_lf[i], i, self.device, self.num_spec)#self.cat_data(self.train_data, self.keys[i], i)
                valid_dict_lf[self.keys_lf[i]], vl_ids, vl_nmols = prep.cat_data(self.valid_data_lf, self.keys_lf[i], i, self.device, self.num_spec)#self.cat_data(self.valid_data, self.keys[i], i)
                test_dict_lf[self.keys_lf[i]], ts_ids, ts_nmols = prep.cat_data(self.test_data_lf, self.keys_lf[i], i, self.device, self.num_spec)#self.cat_data(self.test_data, self.keys[i], i)
            else:
                train_dict_lf[self.keys_lf[i]] = prep.cat_data(self.train_data_lf, self.keys_lf[i], i, self.device, self.num_spec)#self.cat_data(self.train_data, self.keys[i], i)
                valid_dict_lf[self.keys_lf[i]] = prep.cat_data(self.valid_data_lf, self.keys_lf[i], i, self.device, self.num_spec)#self.cat_data(self.valid_data, self.keys[i], i)
                test_dict_lf[self.keys_lf[i]] = prep.cat_data(self.test_data_lf, self.keys_lf[i], i, self.device, self.num_spec)#self.cat_data(self.test_data, self.keys[i], i)


        for epoch in range(max_epochs):
            print('\nGPU ', self.device,' epoch: ', epoch)
            self.openf.write('\nepoch: ' + str(epoch) + '\n')
            self._run_epoch(epoch, train_dict, tr_ids, tr_nmols, tvs='train', data_lf=train_dict_lf)
            self._run_epoch(epoch, valid_dict, vl_ids, vl_nmols, tvs='valid', data_lf=valid_dict_lf)
            if self.device == 0 or self.device == 'cpu':
                self._run_epoch(epoch, test_dict, ts_ids, ts_nmols, tvs='test', data_lf=test_dict_lf)
            
            if self.device == 'cpu' and (epoch % self.save_every == 0 or epoch  + 1 == max_epochs):
                self.save_cpu('model', epoch, self.optimizer)
            if self.device == 0 and (epoch % self.save_every == 0 or epoch  + 1 == max_epochs): 
                self.save_ddp('model', epoch, self.optimizer)

        print('device ', self.device, 'is DONE!!!')
        self.openf.write('Final parameters for device ' +  str(self.device) + ':\n')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.openf.write(name + ': ' + str(param.data) + '\n')
        
        tr1 = time.time()
        self.openf.write('Training time: ' + str(tr1-tr0))
        self.openf.close()



# ====================================================================================================
def spawned_trainer(rank:int, args, world_size:int):
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
    Path(save_path).mkdir(parents=True, exist_ok=True)
    torch.autograd.set_detect_anomaly(True)
    train_set, valid_set, test_set, model, optimizer, lr_scheduler, sae_energies, keys = prep.load_train_objs(args, fid=args.fidlevel, get_aevs=True)
    train_set_lf, valid_set_lf, test_set_lf, keys_lf = prep.load_train_data(args,fid=args.fidlevel_lf, get_aevs=False)
    prep.set_up_task_weights(model, args, optimizer)
    device = rank# if torch.cuda.is_available() else 'cpu'
    if args.ddp:
        model = DDP(model.to(device), device_ids=[device], find_unused_parameters=True)
        model.mtl = model.module.mtl
    else:
        model = model.to(device)
    trainer = Trainer_mf(model, train_set, valid_set, test_set, keys, train_set_lf, valid_set_lf, test_set_lf, keys_lf, optimizer, lr_scheduler, sae_energies, args.save_every, fname, save_path, device, force_train=args.floss, num_spec = args.num_species) 
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
