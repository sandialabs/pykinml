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
import time
import os

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from pykinml import trainer
from pykinml import prepper as prep

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
Next, determine how many nodes/layer and which activation 
function to use for each layer.
The final layer is the output of the neural network.
The length of my_neurons ans my_actfn must be the same.
"""
args.my_neurons = [48, 24, 12, 1]
args.my_actfn = ["gaussian", "gaussian", "gaussian", "identity"]

"""
This code utilizes the atomic envirmoent vectors 
described in the paper by Smith et. al.: https://doi.org/10.1038/sdata.2017.193
Here we set the hyperparameters to build these AEVS. The cutoff radii are in units of Angstroms.
"""
radial_cutoff = 5.2
angular_cutoff = 3.8
args.cuttoff_radius = [radial_cutoff, angular_cutoff]
nrho_rad = 16  # number of radial shells in the radial AEV
nrho_ang = 8  # number of radial shells in the angular AEV
nalpha = 8  # number of angular wedges dividing [0,pi] in the angular AEV
args.aev_params = [nrho_rad, nrho_ang, nalpha]


"""
Now lets look at the input data. pyKinML reads data directly from 
sql files, such as the one in data_holder.
For this example, we also specify a text file saying which structures from the sql file 
to use and which of those structures will be used for training, validation, 
and testing. fidlevel is the fidelity data of the data. We chose 1=wB97X-D/6-311++G(d,p).
Setting args.pre_saved=False lets the model know it should prep the data. If you have already
prepared the dataset (such as for the load_model example), set this to True 
and the data will be loaded without need to prepare it again.
"""
args.input_data_fname = 'data_holder/C5H5.db'      #if training to data from sql, this should be the path to the sql files (readable by glob)
args.input_data_type = 'sqlite'
args.trtsid_name = ["sample_tvt.txt"]
args.fidlevel = 1
args.pre_saved=False

"""
Next let's decide how many points to use for training, validation, and testing. 
In sample_tvt, 2048 structures have a 2 next to them. Those points are set
to be used for either training or validation. ntrain of those, chosen randomly,
will be used for training and nvalid will be used for validation.
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
args.tr_batch_size = 50


"""
Force training is also supported. If you would like to train to both energies ans forces,
set floss to True. fw determines the F:E weight ratio in the loss function. 
The Force weight is fw and the Energy weight is 1-fw. 

Alternativly, you can set optimize_force_weight to true and the F:E weighting in the loss
function will be optimized during the back propagation each training batch. Details of this 
can be found in the paper:
Kendall, Alex, Yarin Gal, and Roberto Cipolla.
"Multi-task learning using uncertainty to weigh losses for scene geometry and semantics."
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018

If floss is set to False, fw and optimize_force_weight are ignored. If optimize_force_weight
is set to True, fw is ignored.
"""
args.floss = False      #set to true to include forces in the loss function
args.fw = 0.0
args.optimize_force_weight = False


"""
Lets set the random seeds used by the model as well as name it.
The inital weights and biases within the model as well as the splitting
of the shuffling of the batches is determined randomly so setting these seeds
allows for repriducibility.
"""
args.randomseed = [0, 1]    #Set the random seeds for initial model parameters and data shuffling. Good for reproducibility
args.savenm = 'my_model_'+str(args.randomseed[0]) + '_'+str(args.randomseed[1])


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
    # set default pytorch as double precision
    torch.set_default_dtype(torch.float64)
    #torch.set_printoptions(precision=8)
    seeds = args.randomseed
    args.num_species = len(args.present_elements)
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    print('args.world_size: ', args.world_size)
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node: ', ngpus_per_node)
    if args.distributed:
        if 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()

    if not args.pre_saved:
        print('preping data')
        prep.prep_data(args, fid=args.fidlevel)
        args.pre_saved = True
        args.data_path = args.savenm

    init_process_group(backend='gloo', init_method='env://',
                       world_size=args.world_size, rank=args.rank)

    save_path = args.savenm +'_gpu_' + str(args.rank) + '.log'
    print('about to load train objects!')
    #save_path = args.savenm + '_seeds_'+str(seeds[0])+'_'+str(seeds[1])
    Path(save_path[:-4]).mkdir(parents=True, exist_ok=True)
    train_set, valid_set, test_set, model, optimizer, lr_scheduler, sae_energies, keys = prep.load_train_objs(args)
    prep.set_up_task_weights(model, args, optimizer)
    model = model.to(args.gpu)
    model = DDP(model.to(args.gpu), device_ids=[args.gpu])
    model.mtl = model.module.mtl
    trainer_loop =  trainer.Trainer(model, train_set, valid_set, test_set, keys, optimizer, lr_scheduler, sae_energies, args.save_every, save_path, args.savenm, args.gpu, force_train=args.floss, num_spec = args.num_species)
    trainer_loop.train(args.epochs)
    destroy_process_group()

if __name__ == "__main__":
    tt1 = time.time()
    main(args)


