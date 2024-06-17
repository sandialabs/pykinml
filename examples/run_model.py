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

import numpy as np
from ase import Atoms
from pykinml import model_calculator as mod

nn = 'my_model_0_1/model-0009.pt'
surr  = mod.Nn_surr(nn, tnsr=True, nrho_rad=16, nrho_ang=8, nalpha=8, R_c=[5.2, 3.8], mf=False, device='cpu')

coord = np.array([[-0.9981252206, -0.2266407005, -0.5827299231],
                     [-0.9349253557, -0.5805118019, 0.6653986964],
                     [0.2299383635, 0.7471148235, -0.9671155914],
                     [0.5032100887, -0.4581358393, 0.8855654072],
                     [1.1736620945, 0.5548549615, 0.0099222856],
                     [-1.7749905917, -0.6584807936, -1.2744784327],
                     [-1.6602525733, -1.2656303199, 1.0401661686],
                     [2.0469469305, 1.2790772568, 0.5610993450],
                     [1.1804663131, -1.1761659084, 1.4914759564],
                     [0.5184324813, 1.3801415798, -1.9407918912]])

spec = ['C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H']

atoms = Atoms(spec, coord)
atoms.set_calculator(surr)
surr.calculate(atoms)
print(surr.results['energy'].item())
print(surr.results['forces'])
