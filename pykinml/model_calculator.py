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
from pykinml import trainer as pes
from pykinml import daev
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

import numpy as np


class Nn_surr(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, fname, restart=None, ignore_bad_restart_file=False, label='surrogate', atoms=None, tnsr=True, nrho_rad=16, nrho_ang=8, nalpha=8, R_c=[5.2, 3.8],
                 **kwargs):
        Calculator.__init__(self, restart=restart, ignore_bad_restart_file=ignore_bad_restart_file, label=label,
                            atoms=atoms, tnsr=tnsr, **kwargs)
        if isinstance(fname, list) and fname.__len__() > 1:
            self.multinn = True
        else:
            self.multinn = False
        self.surrogate = Nnpes_calc(fname, self.multinn)
        self.tnsr = tnsr
        self.nrho_rad = nrho_rad
        self.nrho_ang = nrho_ang
        self.nalpha = nalpha
        self.R_c = R_c

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes, loaddb=None, args=None, xid=None):
        Calculator.calculate(self, atoms, properties, system_changes)
        if 'forces' in properties:
            favail = True
        else:
            favail = False

        xyzd = [[[s for s in atoms.symbols], np.array(atoms.positions)]]
        self.surrogate.dpes.aev_from_xyz(xyzd, self.nrho_rad, self.nrho_ang, self.nalpha, self.R_c, False)
        self.surrogate.nforce = self.surrogate.dpes.full_symb_data[0].__len__() * 3

        if self.multinn:
            energy, Estd, E_hf = self.surrogate.eval()
            if favail:
                force, Fstd, force_ind = self.surrogate.evalforce()
        else:
            energy = self.surrogate.eval()[0][0]
            Estd = torch.tensor(0.)
            if favail:
                force = self.surrogate.evalforce()
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
            self.results['energy'] = energy.detach().numpy()
            self.results['energy_std'] = Estd.detach().numpy()
            if self.multinn:
                self.results['all_energies'] = E_hf.detach().numpy()
            if favail:
                self.results['forces'] = np.reshape(force.detach().numpy(), (-1, 3))
                self.results['forces_std'] = Fstd.detach().numpy()
                if self.multinn:
                    self.results['all_forces'] = force_ind.detach().numpy()

# ====================================================================================================


class My_args():

    def __init__(self, nntype, model_name):
        self.nntype = [nntype]
        if model_name == None:
            self.load_model = False
        else:
            if isinstance(model_name, list):
                self.load_model_name = model_name
            else:
                self.load_model_name = [model_name]
        self.nw = True
        self.savepth = None
        self.savepth_pars = None
        self.device = torch.device('cpu')

class Nnpes_calc():

    def __init__(self, fname, multinn=False):
        self.dpes = data.Data_pes(['C', 'H'])
        if multinn:
            print('Using ensemble')
            self.nmodel = fname.__len__()
            options = [My_args('Comp', fnm) for fnm in fname]
            self.dpes.device = options[0].device
            self.nn_pes = [pes.prep_model(True, opts, load_opt=False) for opts in options]
        else:
            print('Using single model')
            self.nmodel = 1
            options = My_args('Comp', fname)
            self.dpes.device = options.device
            self.nn_pes = pes.prep_model(True, options, load_opt=False)

    def eval(self, indvout=False):
        idl = list(range(0, self.dpes.ndat))
        self.dpes.prep_data(idl)
        self.dpes.indvout = indvout
        self.dpes.xb = [[xt.requires_grad_() for xt in self.dpes.xb[b]] for b in range(self.dpes.nbt)]
        if self.nmodel == 1:
            self.E_lf, self.E_hf = self.nn_pes.eval_dl(self.dpes)
            E_pred = self.E_hf
            return E_pred
        else:
            self.E_hf = torch.empty((self.dpes.ndat, self.nmodel))
            for i in range(self.nmodel):
                E_lf, E_hf = self.nn_pes[i].eval_dl(self.dpes)
                self.E_hf[:, i] = E_hf.reshape(-1)
            E_pred = torch.mean(self.E_hf)
            Estd = torch.std(self.E_hf, 1)
            return E_pred, Estd, self.E_hf

    def evalgrad(self):
        if self.nmodel == 1:
            dEdxyz = daev.cal_dEdxyz_dl(self.dpes, self.E_hf)[0]
            return dEdxyz
        else:
            dEdxyz = torch.empty((self.dpes.ndat, self.nmodel, self.nforce))
            for i in range(self.nmodel):
                tmp = daev.cal_dEdxyz_dl(self.dpes, self.E_hf[:, i])[0]
                dEdxyz[:, i] = tmp
            gmean = torch.mean(dEdxyz, 1).reshape(-1)
            gstd  = torch.std(dEdxyz, 1).reshape(-1)
            return gmean, gstd, dEdxyz

    def evalforce(self):
        if self.nmodel == 1:
            gradient = self.evalgrad()
            return -gradient
        else:
            gmean, gstd, dEdxyz = self.evalgrad()
            return -gmean, gstd, -dEdxyz

