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

import torch
from pykinml import data
import math
from pykinml import trainer as pes
from pykinml import daev as daev
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from io import StringIO



class Nn_surr(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, fname, restart=None, ignore_bad_restart_file=False, label='surrogate', atoms=None, tnsr=True, device='cpu', nrho_rad=16, nrho_ang=8, nalpha=8, R_c=[5.2, 3.8], mf=False, present_elements=['C','H'], 
                 **kwargs):
        Calculator.__init__(self, restart=restart, ignore_bad_restart_file=ignore_bad_restart_file, label=label,
                            atoms=atoms, tnsr=tnsr, **kwargs)
        if isinstance(fname, list) and fname.__len__() > 1:
            self.multinn = True
        else:
            self.multinn = False
        self.surrogate = Nnpes_calc(fname, self.multinn, mf=mf, device=device, present_elements=present_elements)
        self.tnsr = tnsr
        self.device = device
        self.nrho_rad = nrho_rad
        self.nrho_ang = nrho_ang
        self.nalpha = nalpha
        self.R_c = R_c

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes, args=None):
        Calculator.calculate(self, atoms, properties, system_changes)
        if 'forces' in properties:
            favail = True
        else:
            favail = False
        if atoms is None:
            atoms = self.atoms
        xyzd = [[[s for s in atoms.symbols], np.array(atoms.positions)]]
        self.surrogate.dpes.aev_from_xyz(xyzd, self.nrho_rad, self.nrho_ang, self.nalpha, self.R_c, False)
        self.surrogate.nforce = self.surrogate.dpes.full_symb_data[0].__len__() * 3

        if self.multinn:
            energy, Estd, E_hf = self.surrogate.eval()
            if favail:
                force, Fstd, force_ind = self.surrogate.eval_force()
        else:
            energy = self.surrogate.eval()[0][0]
            Estd = torch.tensor(0.)
            if favail:
                force = self.surrogate.eval_force()
                Fstd = torch.tensor(0.)

        if self.tnsr:
            self.results['energy'] = energy
            self.results['energy_std'] = Estd
            if self.multinn:
                self.results['all_energies'] = E_hf
            if favail:
                self.results['forces'] = force.view(-1, 3)
                self.results['forces_std'] = Fstd
                if self.multinn:
                    self.results['all_forces'] = force_ind
        else:
            if self.device=='cpu':
                self.results['energy'] = energy.detach().numpy()
                self.results['energy_std'] = Estd.detach().numpy()
                if self.multinn:
                    self.results['all_energies'] = E_hf.detach().numpy()
                if favail:
                    self.results['forces'] = np.reshape(force.detach().numpy(), (-1, 3))
                    self.results['forces_std'] = Fstd.detach().numpy()
                    if self.multinn:
                        self.results['all_forces'] = force_ind.detach().numpy()
            else:
                self.results['energy'] = energy.detach().cpu().numpy()
                self.results['energy_std'] = Estd.detach().cpu().numpy()
                if self.multinn:
                    self.results['all_energies'] = E_hf.detach().cpu().numpy()
                if favail:
                    self.results['forces'] = np.reshape(force.detach().cpu().numpy(), (-1, 3))
                    self.results['forces_std'] = Fstd.detach().cpu().numpy()
                    if self.multinn:
                        self.results['all_forces'] = force_ind.detach().cpu().numpy()


# ====================================================================================================

class My_args():

    def __init__(self, load_model_name, mf=False, present_elements=['C','H']):
        self.load_model_name = load_model_name
        self.multi_fid = mf
        self.num_species = len(present_elements)
        self.present_elements=present_elements

# ==============================================================================================
class Nnpes_calc():

    def __init__(self, fname, multinn=False, device='cpu', mf=False, present_elements=['C','H']):
        self.device = device
        self.dpes = data.Data_pes(atom_types=present_elements)
        self.present_elements=present_elements
        if multinn:
            print('MULTINET!!!')
            args_list = [My_args(fname[i], mf=mf, present_elements=present_elements) for i in range(len(fname))]
            #self.nmodel = fname.__len__()
            self.nn_pes = [pes.Runner(args_list[i], self.device) for i in range(len(fname))]
        else:
            args = My_args(fname, mf=mf, present_elements=present_elements)
            self.nmodel = 1
            self.nn_pes = pes.Runner(args, self.device)

    def eval(self, indvout=False):
        self.dpes.prep_data(device=self.device)
        self.aevs = [aev[j].requires_grad_() for j in range(len(self.present_elements)) for aev in self.dpes.aevs]
        if self.nmodel == 1:
            self.E = self.nn_pes.eval(self.dpes.aevs)
            E_pred = self.E
            return E_pred
        else:
            self.E = torch.empty((self.dpes.ndat, self.nmodel))
            for i in range(self.nmodel):
                E = self.nn_pes[i].eval(self.dpes.aevs)
                self.E[:, i] = E.reshape(-1)
            E_pred = torch.mean(self.E)
            Estd = torch.std(self.E, 1)
            return E_pred, Estd, self.E

    def eval_force(self):
        if self.nmodel == 1:
            forces = self.nn_pes.eval_force(self.dpes.daevs)
            return forces[0][0]
        else:
            Forces = torch.empty((self.dpes.ndat, self.nmodel, self.nforce))
            for i in range(self.nmodel):
                forces = self.nn_pes[i].eval_force(self.dpes.daevs)
                Forces[:,i] = forces[0][0]
            Fmean = torch.mean(Forces, 1).reshape(-1)
            Fstd  = torch.std(Forces, 1).reshape(-1)
            return Fmean, Fstd, Forces


