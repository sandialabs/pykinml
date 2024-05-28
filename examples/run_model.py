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
from ase.io import read
from pykinml import  model_calculator 
from pykinml.model_calculator import Nn_surr, Nnpes_calc






# set default pytorch as double precision
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=12)




e1 = '../ensemble/comp_r0-0150.pt'
e2 = '../ensemble/comp_r2-0150.pt'
e3 = '../ensemble/comp_r4-0150.pt'
e4 = '../ensemble/comp_r6-0150.pt'
e5 = '../ensemble/comp_r8-0150.pt'

fn_nn = [e1, e2, e3, e4, e5]
surr  = Nn_surr(fn_nn,tnsr=False, nrho_rad=16, nrho_ang=8, nalpha=8, R_c=[5.2, 3.8])
    
mol = read('c5h5.xyz')
print(mol)
mol.set_calculator(surr)
surr.calculate(mol)
eng = surr.results['energy'].item()
std = surr.results['energy_std'].item()
force = surr.results['forces']

print('energy: ', eng)
print('standard deviation: ', std)
print('forces: ', force)



