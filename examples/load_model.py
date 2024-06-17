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

from pykinml import trainer
import torch
import torch.multiprocessing as mp


import time
import os
from pykinml import prepper as prep

args = prep.parse_arguments_list()


"""
First, decide how many epochs you want to train
and how often to save the model.
"""
args.epochs = 10
args.save_every = 1

"""
Then set the parameters to load the model.
savenm is the name of the new model.
data_path is where the data should be saved if pre_saved is set to True.
load_model_name is the path to the model you are loading.
load_opt tells the model to load the optimizer in addition
to the model. If you want to use a different optimizer,
set args.optimizer.  The Adam optimizer is the default option and 
was the optimizer used by train_single_fid.
pre_saved tells the model the data you want to train to
has already been saved.
"""
args.savenm = 'phase_two'
args.data_path = 'my_model_0_1/'
args.load_model = True
args.load_model_name = 'my_model_0_1/model-0009.pt'
args.load_opt = True
args.pre_saved = True

"""
To use the same training/validation/test set that was used in the inital training phase,
set pre_saved=True and use the same numbers for ntrain, nvalid, and ntest.
A different training set can be used, but for this example, will will resuse the same one.
Also, if you want the dataset to be shuffled the same way,
make sure that the random seeds are the same or the shuffling of the training
and validation set will be different. For consistancy, we set tr_batch_size to the same value as in 
train_single_fid.
"""
args.randomseed = [0, 1]    #Set the random seeds for initial model parameters and data shuffling. Good for reproducibility
ntrain = 2000  #number of training structures
nvalid = 48    #number of validation structures
ntest = 0    #number of test structures. When reading tvt mask from a text file, ntest should always be set to 0. The test ids will be read from the text.
args.tvt = [ntrain, nvalid, ntest]
args.tr_batch_size=50

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
args.fw = 0.0
args.optimize_force_weight = False

"""
Finally, we set ddp (Distributed Data Parallel) to True or False
If True, training will be performed on the gpus with indexes 
specified in args.gpus.
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
        prep.prep_data(args, fid=args.fidlevel)
        args.pre_saved = True
    world_size = torch.cuda.device_count()
    print('world_size: ', world_size)
    if args.ddp:
        mp.spawn(trainer.spawned_trainer, args=[args,world_size], nprocs=world_size, join=True)
    else:
        trainer.spawned_trainer('cpu', args, 0)
    tt2 = time.time()
    print('Time it took: ', tt2-tt1)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main(args)



