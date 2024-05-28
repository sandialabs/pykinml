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

def normalize(x, mx, mn):
    d = mx - mn
    if isinstance(x, float):
        newx = (x - mn) / d
    else:
        if len(x[0]) > 1:
            newx = [(xx - mn)/d for xx in x]
        else:
            newx = (x - mn)/d #* nmx + nmn
    return newx

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

def fitted_softplus(input, alpha=0.1, beta=20):
    """Softplus function parametrized to be equal to a CELU
    This allows keeping the good characteristics of CELU, while having an
    infinitely differentiable function.
    It is highly recommended to leave alpha and beta as their defaults,
    which match closely CELU with alpha = 0.1"""
    return torch.nn.functional.softplus(input + alpha, beta=beta) - alpha


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

class simpleNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(simpleNet, self).__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            for i in range(out_dim):
                self.fc.bias[i] = 0
                for j in range(in_dim):
                    self.fc.weight[i, j] = i + j + 1.0
        print("Net weights:\n", self.fc.weight)
        print("Net biases :\n", self.fc.bias)

    def forward(self, x):
        y = self.fc(x)
        return y


class simpleNetsq(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(simpleNetsq, self).__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            for i in range(out_dim):
                self.fc.bias[i] = 0
                for j in range(in_dim):
                    self.fc.weight[i, j] = i + j + 1.0
        print("Net weights:\n", self.fc.weight)
        print("Net biases :\n", self.fc.bias)

    def forward(self, x):
        y = square(self.fc(x))
        return y

class FullyConnectedNet(nn.Module):

    def __init__(self, input_size, neurons, activations):
        super(FullyConnectedNet, self).__init__()

        # For now, we will have a linear layer followed by an activation function
        assert len(neurons) == len(activations), 'Number of neurons must be equal to the number of activations'

        # We will need a list of blocks cascaded one after the other, so we keep them in a ModuleList instead of a Python list
        self.blocks = nn.ModuleList()

        previous = input_size
        for i in range(len(neurons)):
            self.blocks.append(LinearBlock(previous, neurons[i], activations[i]))
            previous = neurons[i]

    def forward(self, x):
        #Pass the input through each block
        for block in self.blocks:
            x = block(x)

        return x

class SingleNetwork_lf(nn.Module):
    def __init__(self, n_input_a, n_input_b, neurons_a, activations_a, neurons_b, activations_b):
        super(SingleNetwork_lf, self).__init__()

        # For now, we will have a linear layer followed by an activation function
        assert len(neurons_a) == len(activations_a), \
            'NN_a: Number of neurons must be equal to the number of activations'
        assert len(neurons_b) == len(activations_b), \
            'NN_b: Number of neurons must be equal to the number of activations'

        # We will need a list of blocks cascaded one after the other,
        # so we keep them in a ModuleList instead of a Python list
        self.NNa = nn.ModuleList()
        self.NNb = nn.ModuleList()

        previous_a = n_input_a
        for i in range(len(neurons_a)):
            self.NNa.append(LinearBlock(previous_a, neurons_a[i], activations_a[i]))
            previous_a = neurons_a[i]

        previous_b = n_input_b
        for j in range(len(neurons_b)):
            self.NNb.append(LinearBlock(previous_b, neurons_b[j], activations_b[j]))
            previous_b = neurons_b[j]

    def forward(self, Xa, Xb, idx, **kwargs):

        for block_a in self.NNa:
            Xa = block_a(Xa)
        out_a = Xa

        for block_b in self.NNb:
            Xb = block_b(Xb)
        out_b = Xb

        out = torch.sum(
            torch.stack(
                [segment_coo(out_a, idx[0], reduce="sum"), segment_coo(out_b, idx[1], reduce="sum")], 0),
            dim=0)

        return out_a, out_b, out


class SingleNetwork_hf(nn.Module):
    def __init__(self, n_input, neurons, activations):
        super(SingleNetwork_hf, self).__init__()

        # For now, we will have a linear layer followed by an activation function
        assert len(neurons) == len(activations), \
            'NN: Number of neurons must be equal to the number of activations'

        # We will need a list of blocks cascaded one after the other,
        # so we keep them in a ModuleList instead of a Python list
        self.NN = nn.ModuleList()

        previous = n_input
        for i in range(len(neurons)):
            self.NN.append(LinearBlock(previous, neurons[i], activations[i]))
            previous = neurons[i]

    def forward(self, X, **kwargs):

        for block in self.NN:
            X = block(X)
        out = X
        return out


class CompositeNetworks(nn.Module):
    def __init__(self, arch, n_input_a, n_input_b, neurons_a, activations_a, neurons_b, activations_b, neurons_hf=None,
                 activations_hf=None, neurons_hf_b=None, activations_hf_b=None, sae_energies=[0, 0], add_sae=False, biases=True):
        super(CompositeNetworks, self).__init__()

        self.arch = arch
        # For now, we will have a linear layer followed by an activation function
        assert len(neurons_a) == len(activations_a), \
            'NN_a: Number of neurons must be equal to the number of activations'
        assert len(neurons_b) == len(activations_b), \
            'NN_b: Number of neurons must be equal to the number of activations'

        # We will need a list of blocks cascaded one after the other,
        # so we keep them in a ModuleList instead of a Python list
        self.NNa = nn.ModuleList()
        self.NNb = nn.ModuleList()

        previous_a = n_input_a
        for i in range(len(neurons_a)):
            self.NNa.append(LinearBlock(previous_a, neurons_a[i], activations_a[i], biases=biases))
            previous_a = neurons_a[i]

        previous_b = n_input_b
        for j in range(len(neurons_b)):
            self.NNb.append(LinearBlock(previous_b, neurons_b[j], activations_b[j], biases=biases))
            previous_b = neurons_b[j]

        self.sae_energies = sae_energies
        #self.sae_energies_lf = sae_energies_lf
        self.add_sae = add_sae
        if arch != 'hfonly':
            self.NN_hf = nn.ModuleList()

            if arch == 'seq-2net':
                self.NN_hf_b = nn.ModuleList()
                previous_hf = previous_a
                for i in range(len(neurons_hf)):
                    self.NN_hf.append(LinearBlock(previous_hf, neurons_hf[i], activations_hf[i]))
                    previous_hf = neurons_hf[i]

                previous_hf_b = previous_b
                for j in range(len(neurons_hf_b)):
                    self.NN_hf_b.append(LinearBlock(previous_hf_b, neurons_hf_b[j], activations_hf_b[j]))
                    previous_hf_b = neurons_hf_b[j]


            elif arch == 'seq-1net':
                previous_hf = 1
                for i in range(len(neurons_hf)):
                    self.NN_hf.append(LinearBlock(previous_hf, neurons_hf[i], activations_hf[i]))
                    previous_hf = neurons_hf[i]

            elif arch == 'hybrid-seq-2net':
                self.NN_hf_b = nn.ModuleList()
                previous_hf = previous_a + n_input_a
                for i in range(len(neurons_hf)):
                    self.NN_hf.append(LinearBlock(previous_hf, neurons_hf[i], activations_hf[i]))
                    previous_hf = neurons_hf[i]

                previous_hf_b = previous_b + n_input_b
                for j in range(len(neurons_hf_b)):
                    self.NN_hf_b.append(LinearBlock(previous_hf_b, neurons_hf_b[j], activations_hf_b[j]))
                    previous_hf_b = neurons_hf_b[j]


            elif arch == 'hybrid-seq-1net':
                print('Single output of low-fid NN cannot be combined with the original input data.')

            elif arch == 'hybrid-dscr':
                self.NN_hf_b = nn.ModuleList()
                previous_hf = n_input_a + previous_a
                for i in range(len(neurons_hf)):
                    self.NN_hf.append(LinearBlock(previous_hf, neurons_hf[i], activations_hf[i]))
                    previous_hf = neurons_hf[i]

                previous_hf_b = n_input_b + previous_b
                for j in range(len(neurons_hf_b)):
                    self.NN_hf_b.append(LinearBlock(previous_hf_b, neurons_hf_b[j], activations_hf_b[j]))
                    previous_hf_b = neurons_hf_b[j]

            elif arch == 'dscr':
                self.NN_hf_b = nn.ModuleList()
                previous_hf = n_input_a
                for i in range(len(neurons_hf)):
                    self.NN_hf.append(LinearBlock(previous_hf, neurons_hf[i], activations_hf[i]))
                    previous_hf = neurons_hf[i]

                previous_hf_b = n_input_b
                for j in range(len(neurons_hf_b)):
                    self.NN_hf_b.append(LinearBlock(previous_hf_b, neurons_hf_b[j], activations_hf_b[j]))
                    previous_hf_b = neurons_hf_b[j]

    def forward(self, Xa, Xb, idx, Xhfa=None, Xhfb=None, idx_hf=None, indvout=False, maxmin=[1., 0.], **kwargs):
       
        #print('Xa.shape: ', Xa.shape)

        
        if len(idx[0])>0:
            for block_a in self.NNa:
                Xa = block_a(Xa)
            out_a = Xa
            if self.add_sae:
                out_a += self.sae_energies[0]
            seg_a = segment_coo(out_a, idx[0], reduce="sum")
        else:
            out_a = torch.zeros(1,1)
            seg_a = segment_coo(out_a, torch.tensor([0]), reduce="sum")

        if len(idx[1])>0:
            for block_b in self.NNb:
                Xb = block_b(Xb)
            out_b = Xb
            if self.add_sae:
                out_b += self.sae_energies[1]
            seg_b = segment_coo(out_b, idx[1], reduce="sum")
        else:
            out_b = torch.zeros(1,1)
            seg_b = segment_coo(out_b, torch.tensor([0]), reduce="sum")
        
        if self.arch == 'hfonly':
            out_C = out_a
            out_H = out_b
            #out = torch.sum(torch.stack([segment_coo(out_a, idx[0], reduce="sum"),
            #                             segment_coo(out_b, idx[1]), reduce="sum")], 0), dim=0)
            out = torch.sum(torch.stack([seg_a, seg_b], 0), dim=0)
            out_lf = None

        else:
            Xlfa = Xhfa.clone()
            Xlfb = Xhfb.clone()
            for block_a in self.NNa:
                Xlfa = block_a(Xlfa)
            out_lf_a = Xlfa
            for block_b in self.NNb:
                Xlfb = block_b(Xlfb)
            out_lf_b = Xlfb

            out_lf_C = out_a
            out_lf_H = out_b
            out_lf = torch.sum(torch.stack([segment_coo(out_a, idx[0], reduce="sum"),
                                            segment_coo(out_b, idx[1], reduce="sum")], 0), dim=0)

            if self.arch == 'seq-2net':
                for block_hf in self.NN_hf:
                    out_lf_a = block_hf(normalize(out_a, *maxmin))
                out_hf_a = out_lf_a

                for block_hf_b in self.NN_hf_b:
                    out_lf_b = block_hf_b(normalize(out_b, *maxmin))
                out_hf_b = out_lf_b

                out_C = out_hf_a
                out_H = out_hf_b
                out = torch.sum(torch.stack([segment_coo(out_hf_a, idx_hf[0], reduce="sum"),
                                             segment_coo(out_hf_b, idx_hf[1], reduce="sum")], 0), dim=0)

            elif self.arch == 'seq-1net':
                out_lf_C = out_lf_a
                out_lf_H = out_lf_b
                Xhf = torch.sum(torch.stack([segment_coo(out_lf_a, idx_hf[0], reduce="sum"),
                                             segment_coo(out_lf_b, idx_hf[1], reduce="sum")], 0), dim=0)

                for block_hf in self.NN_hf:
                    Xhf = block_hf(normalize(Xhf, *maxmin))
                out = Xhf
                if indvout:
                    out_C = None
                    out_H = None

            elif self.arch == 'hybrid-seq-2net':
                Xhfa2 = torch.cat((Xhfa, normalize(out_lf_a, *maxmin)), 1)
                Xhfb2 = torch.cat((Xhfb, normalize(out_lf_b, *maxmin)), 1)
                for block_hf in self.NN_hf:
                    Xhfa2 = block_hf(Xhfa2)
                out_hf_a = Xhfa2
                if self.add_sae:
                    out_hf_a += self.sae_energies[0]
                for block_hf_b in self.NN_hf_b:
                    Xhfb2 = block_hf_b(Xhfb2)
                out_hf_b = Xhfb2
                if self.add_sae:
                    out_hf_b += self.sae_energies[0]
                out_C = out_hf_a
                out_H = out_hf_b
                out = torch.sum(torch.stack([segment_coo(out_hf_a, idx_hf[0], reduce="sum"),
                                             segment_coo(out_hf_b, idx_hf[1], reduce="sum")], 0), dim=0)

            elif self.arch == 'hybrid-dscr':
                Xhfa2 = torch.cat((Xhfa, normalize(out_lf_a, *maxmin)), 1)
                for block_hf in self.NN_hf:
                    Xhfa2 = block_hf(Xhfa2)
                out_hf_a = Xhfa2
                if self.add_sae:
                    out_hf_a += self.sae_energies[0]

                Xhfb2 = torch.cat((Xhfb, normalize(out_lf_b, *maxmin)), 1)
                for block_hf_b in self.NN_hf_b:
                    Xhfb2 = block_hf_b(Xhfb2)
                out_hf_b = Xhfb2
                if self.add_sae:
                    out_hf_b += self.sae_energies[0]
                out_C = out_hf_a
                out_H = out_hf_b
                out_hf = torch.sum(torch.stack([segment_coo(out_hf_a, idx_hf[0], reduce="sum"),
                                                segment_coo(out_hf_b, idx_hf[1], reduce="sum")], 0), dim=0)

                out_lf2 = torch.sum(
                    torch.stack(
                        [segment_coo(out_lf_a, idx_hf[0], reduce="sum"),
                         segment_coo(out_lf_b, idx_hf[1], reduce="sum")], 0),
                    dim=0)

                out = out_lf2 + out_hf

            elif self.arch == 'dscr':
                for block_hf in self.NN_hf:
                    Xhfa = block_hf(Xhfa)
                out_hf_a = Xhfa

                for block_hf_b in self.NN_hf_b:
                    Xhfb = block_hf_b(Xhfb)
                out_hf_b = Xhfb

                out_C = out_hf_a
                out_H = out_hf_b
                out_hf = torch.sum(torch.stack([segment_coo(out_hf_a, idx_hf[0], reduce="sum"),
                                                segment_coo(out_hf_b, idx_hf[1], reduce="sum")], 0), dim=0)

                out_lf2 = torch.sum(
                    torch.stack(
                        [segment_coo(out_lf_a, idx_hf[0], reduce="sum"),
                         segment_coo(out_lf_b, idx_hf[1], reduce="sum")], 0),
                    dim=0)

                out = out_lf2 + out_hf
        if indvout:
            return out_lf, out, out_C, out_H
        else:
            return out_lf, out

