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
import torch.nn as nn
from torch_scatter import segment_coo
import os


#=======================================================================
# define an identity activation function
def identity(input):
    '''
    Applies an identity activation function (i.e. no activation) element-wise:
        identity(x) = x
    '''
    return input

#=======================================================================
# define a gaussian activation function
def gaussian(input):
    '''
    Applies the Gaussian activation function element-wise:
        gaussian(x) = exp(-x^2)
    '''
    return torch.exp(-input**2)

#=======================================================================
# define a x2 function
def square(input):
    return input**2

#=======================================================================
# define my ReLU function
def my_relu(input):
    return torch.relu(input)

#=======================================================================
# define a silu function
def silu(input):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x) = x * 1/(1+exp(-x))
    '''
    return input * torch.sigmoid(input)



class LinearBlock(nn.Module):
    def __init__(self, in_nums, out_nums, activation, biases=True):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_nums, out_nums, bias=biases)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.linear(x))



class CompositeNetworks(nn.Module):
    def __init__(self, n_input, neurons, activations, neurons_hf=None,
                 activations_hf=None, neurons_hf_b=None, activations_hf_b=None, sae_energies=[0, 0], add_sae=False, biases=True):
        super(CompositeNetworks, self).__init__()

        
        self.NNs = []
        for i in range(len(neurons)):
            NN = nn.ModuleList()
            previous = n_input[i]
            for j in range(len(neurons[i])):
                NN.append((LinearBlock(previous, neurons[i][j], activations[i][j], biases=biases)))
                previous = neurons[i][j]
            self.NNs.append(NN)
        self.NNs = nn.ParameterList(self.NNs)

        
       
        self.add_sae = add_sae
        self.sae_energies = sae_energies


    def forward(self, Xs, inds, nmols=1, train=True):
        segments = []
        for X in range(len(Xs)):
            if len(inds[X]) > 0:
                lo = Xs[X]
                for block in self.NNs[X]:
                    lo = block(lo)
                if self.add_sae:
                    lo += self.sae_energies[X]
                segments.append(segment_coo(lo, inds[X], dim_size=nmols, reduce="sum"))
            else:
                segments.append(torch.zeros(nmols,1, device=Xs[X].device))
                #segments.append(torch.zeros(nmols,1))
        out = torch.sum(torch.stack(segments), dim=0)
        
        #log_sigma = 1.0 * self.log_sigma

        if train:
            log_sigma = 1.0 * self.log_sigma
            return out, log_sigma
        else:
            return out


class CompositeNetworks_MF(nn.Module):
    def __init__(self, n_input, neurons, activations, neurons_hf=None,
                 activations_hf=None, sae_energies=[0, 0], add_sae=False, biases=True, dscr=True, pass_eng=True, pass_aev=True):
        super(CompositeNetworks_MF, self).__init__()

        if not pass_eng and not pass_aev:
            print('Both pass_eng and pass_aev are set to False')
            print('Must use either previous LF output, AEV, or both as input to HF network!')
            print('Try setting pass_eng=True or pass_aev=True (or both)')
            sys.exit()

        self.pass_aev = pass_aev
        self.pass_eng = pass_eng
        self.dscr = dscr
        self.NNs = []
        for i in range(len(neurons)):
            NN = nn.ModuleList()
            previous = n_input[i]
            for j in range(len(neurons[i])):
                NN.append((LinearBlock(previous, neurons[i][j], activations[i][j], biases=biases)))
                previous = neurons[i][j]
            self.NNs.append(NN)
        self.NNs = nn.ParameterList(self.NNs)

        self.NNs_hf = []
        for i in range(len(neurons_hf)):
            NN_hf = nn.ModuleList()
            if self.pass_eng:
                previous = neurons[i][-1]
                if self.pass_aev:
                    previous += n_input[i]
            else:
                previous = n_input[i]

            for j in range(len(neurons_hf[i])):
                NN_hf.append((LinearBlock(previous, neurons_hf[i][j], activations_hf[i][j], biases=biases)))
                previous = neurons_hf[i][j]
            self.NNs_hf.append(NN_hf)
        self.NNs_hf = nn.ParameterList(self.NNs_hf)


        self.add_sae = add_sae
        self.sae_energies = sae_energies


    def forward(self, Xs, inds, nmols=1, maxmin=[1., 0.], train=True):
        segments_lf = []
        segments_hf = []
        for X in range(len(Xs)):
            if len(inds[X]) > 0:
                lo = Xs[X]
                hi = Xs[X]
                for block in self.NNs[X]:
                    lo = block(lo)
                if self.add_sae:
                    lo += self.sae_energies[X]
                segments_lf.append(segment_coo(lo, inds[X], dim_size=nmols, reduce="sum"))
                if self.pass_eng:
                    hi = lo
                    if self.pass_aev:
                        hi = torch.cat((Xs[X], hi), dim=1)
                else:
                    hi = Xs[X]
                for block in self.NNs_hf[X]:
                    hi = block(hi)
                segments_hf.append(segment_coo(hi, inds[X], dim_size=nmols, reduce='sum'))


            else:
                segments_lf.append(torch.zeros(nmols,1))
                segments_hf.append(torch.zeros(nmols,1))
        out_lf = torch.sum(torch.stack(segments_lf), dim=0)
        out_hf = torch.sum(torch.stack(segments_hf), dim=0)# + out
        if self.dscr:
            out_hf += out_lf

        
        if train:
            log_sigma = 1.0 * self.log_sigma
            return out_lf, out_hf, log_sigma
        else:
            return out_lf, out_hf

