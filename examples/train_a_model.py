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

import json
from pykinml import trainer
from pykinml.trainer import train_pes
import torch



class My_args():
    def __init__(self, args):
        for key, value in data.items():
            if value=='False':
                setattr(self, key, False)
            elif value=='True':
                setattr(self, key, True)
            elif value=='None':
                setattr(self, key, None)
            else:
                setattr(self, key, value)



"""
First we load the arguments from args.json. 
Explanations for each of the arguments can be 
found in trainer.py in the parse_arguments_list 
function but we will examine a few here.
"""

f = open('args.json')

# returns JSON object as
# a dictionary
data = json.load(f)
print(data)
args = My_args(data)


"""
First, decide how many epochs you want to train
and how often to save the model.
"""
args.epochs = 5
args.save_every = 1

"""
Next, determine how many nodes/layer and which activation 
function to use for each layer.
The final layer is the output of the neural network.
The length of my_neurons ans my_actfn must be the same.
"""
args.my_neurons = [12, 6, 2, 1]
args.my_actfn = ["gaussian", "gaussian", "gaussian", "identity"]

"""
This code utilizes the atomic envirmoent vectors 
described in the paper by Smith et. al.: https://doi.org/10.1038/sdata.2017.193
Here we set the hyperparameters to build these AEVS.
"""
radial_cutoff = 5.2
angular_cutoff = 3.8
args.cuttoff_radius = [radial_cutoff, angular_cutoff]
nrho_rad = 4  # number of radial shells in the radial AEV
nrho_ang = 2  # number of radial shells in the angular AEV
nalpha = 2  # number of angular wedges dividing [0,pi] in the angular AEV
args.aev_params = [nrho_rad, nrho_ang, nalpha]


"""
Now lets look at the input data. trainer.py can read data directly from 
sql files, such as the one in data_holder, or from a prepackaged hdf file.
We also specify a text file saying which structures from the sql file 
to use and which of those structures will be used for training, validation, 
and testing. If you want to generate an hdf file to train to in the future,
set write_hdf = True
"""
args.input_data_fname = ['data_holder/C5H5.db']      #if training to data from sql, this should be the path to the sql files (readable by glob). If training to data from an hdf, this should be the path to the hdf file.
args.input_data_type = 'sqlite'   #sqlite or aev. aev indicates to train using the aev data stored in an hdf file.
args.trtsid_name = ["sample_tvt.txt"]
args.write_hdf = False     #Set to true to generate an hdf file with training/validation/test data

"""
Next let's decide how many points to use for training, validation, and testing. 
In sample_tvt, 2048 structures have a 2 next to them. Those points are set
to be used for either training or validation. ntrain of those, chosen randomly
in data.py, will be used for training and nvalid will be used for validation.
There is no overlap between the training and the validation set so now single structure
will be used in both. This also means that if ntrain + nvalid is greater than the 
number of structures in sample_tvt that are designated for training/validation, 
an error will occur and the model will not train.

ntest is set to 0. When using a txt file to designate which molecules are used for
trainng, which for validation, and which for testing, this ntest should
always be set to 0 as the test set will be read from the file.
We will also set a training batch size. If ntrain/tr_batch_size is not a
whole number some batches will be smaller.
"""
ntrain = 2000  #number of training structures
nvalid = 48    #number of validation structures
ntest = 0    #number of test structures. When reading tvt mask from a text file, ntest should always be set to 0. The test ids will be read from the text.
args.tvt = [ntrain, nvalid, ntest]
args.tr_batch_size = 5


"""
Force training is also supported. If you would like to train to both energies ans forces,
set floss to True. fw determines the F:E weight ratio in the loss function. 
The Force weight is fw and the Energy weight is 1-fw. 

Alternativly, you can set optimize_force_weight to true and the F:E weighting in the loss
function will be optimized during the back propigation each training batch. Details of this 
can be found in the paper:
Kendall, Alex, Yarin Gal, and Roberto Cipolla.
"Multi-task learning using uncertainty to weigh losses for scene geometry and semantics."
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018

If floss is set to False, fw and optimize_force_weight are ignored. If optimize_force_weight
is set to True, fw is ignored.
"""
args.floss = False      #set to true to include forces in the loss function
args.fw = 0.5
args.optimize_force_weight = False


"""
Finally, lets set the random seeds used by the model as well as name it.
The inital weights and biases within the model as well as the splitting
of the shuffling of the batches is determined randomly so setting these seeds
allows for repriducibility.
"""
args.randomseed = [0, 1]    #Set the random seeds for initial model parameters and data shuffling. Good for reproducibility
args.savenm = ['my_model_'+str(args.randomseed[0]) + '_'+str(args.randomseed[1])]


"""
If you would like to run on a GPU set
enable_cuda=True
If cuda is available available, the training will commence on a GPU,
otherwise it will run on a CPU.
"""
args.enable_cuda = False


args.device = None
if args.enable_cuda and torch.cuda.is_available():
    if args.node == 0:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cuda:'+str(args.node))
else:
    args.device = torch.device('cpu')


"""
Let the training begin!
"""
train_pes(args)
