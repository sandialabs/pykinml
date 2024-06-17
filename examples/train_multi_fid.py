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

import time
import os

import torch
import torch.multiprocessing as mp

from pykinml import prepper as prep
from pykinml import multi_fid_loop as pes_mf



"""
Before training we prep.parse_arguments_list(). This initializes the necessary arguments
to thier default values. Many of these areguments are specific to only one training type
but we will manually set the most relevent ones here.
"""
args = prep.parse_arguments_list()

"""
First, decide how many epochs you want to train
and how often to save the model.
"""
args.epochs = 10
args.save_every = 1

"""
Then lets set up the Multi Fidelity part. fidlevel is the higher (more accurate) 
of the two fidelity levels. fidlevel_lf is the lower fidelity level. In the sample 
dataset we are using, fidelity 1 is wb97x/6-311++G(d,p) and fidelity 0 is b3lyp/6-31G*
"""
args.multi_fid = True
args.fidlevel = 1
args.fidlevel_lf = 0
args.sae_fit=False

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

It is important to note that for multi fidelity training each data point must
have the energies (and forces if floss=True) at both fidelity levels present
in the data file.
"""
args.input_data_fname = 'data_holder/C[2-3]H[2-3].db'      #if training to data from sql, this should be the path to the sql files (readable by glob)
args.input_data_type = 'sqlite'
args.trtsid_name = ["mf_tvt.txt"]

"""
Next let's decide how many points to use for training, validation, and testing. 
In mf_tvt, 40 structures have a 2 next to them. Those points are set
to be used for either training or validation. ntrain of those, chosen randomly,
will be used for training and nvalid will be used for validation.
There is no overlap between the training and the validation set so no single structure
will be used in both. This also means that if ntrain + nvalid is greater than the 
number of structures in mf_tvt that are designated for training/validation, 
an error will occur and the model will not train.

ntest is set to 0. When using a txt file to designate which molecules are used for
trainng, which for validation, and which for testing, this ntest should
be set to 0 as the test set will be read from the file.
We will also set a training batch size. If ntrain/tr_batch_size is not a
whole number some batches will be smaller.
"""
ntrain = 30  #number of training structures
nvalid = 10    #number of validation structures
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
args.floss = True      #set to true to include forces in the loss function
args.fw = 0.5
args.optimize_force_weight = False


"""
Lets set the random seeds used by the model as well as name it.
The inital weights and biases within the model as well as the splitting
of the shuffling of the batches is determined randomly so setting these seeds
allows for repriducibility.
"""
args.randomseed = [0, 1]    #Set the random seeds for initial model parameters and data shuffling. Good for reproducibility
args.savenm = 'mf_model_'+str(args.randomseed[0]) + '_'+str(args.randomseed[1])

"""
Finally, we set ddp (Distributed Data Parallel) to True or False
If True, training will be performed on the gpus with indexes 
specified in args.gpus. If False, args.gpus will be ignored.
"""
args.ddp = False
args.gpus = [0, 1]

"""
Let the training begin!
"""

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
        prep.prep_data(args, fid=args.fidlevel, get_aevs=True)
        prep.prep_data(args, fid=args.fidlevel_lf, get_aevs=False)
        args.pre_saved = True
        args.data_path = args.savenm
    world_size = torch.cuda.device_count()
    print('world_size: ', world_size)
    if args.ddp:
        mp.spawn(pes_mf.spawned_trainer, args=[args,world_size], nprocs=world_size, join=True)
    else:
        pes_mf.spawned_trainer('cpu', args, 0)
    tt2 = time.time()
    print('Time it took: ', tt2-tt1)




if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main(args)



