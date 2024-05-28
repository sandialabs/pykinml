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

import math
import sys
import time
import os
import random
import glob
import pickle
import timeit
import numpy as np

import torch
from ase import Atoms


from pykinml import hdf5_handler as hd
from pykinml import aev
from pykinml import rdb
try:
    import aevmod
except ModuleNotFoundError:
    pass


verbose  = False
newhdfg  = True




# ====================================================================================================
def parse_meta_db(meta_db):
    nblk = len(meta_db)
    meta_parsed = []
    for blk in range(0, nblk):
        meta_parsed.append(parse_meta(meta_db[blk]))
        #print('in parse_meta_db: meta_parsed:',meta_parsed[0])
        #sys.exit('debug exit')
    return meta_parsed


def parse_meta(meta):
    meta_xyz = None
    meta_energy = None
    meta_method = None
    meta_type = None
    meta_name = None
    meta_path = None
    meta_force = None
    meta_name = None
    skip = 0
    if len(meta) == 1:
        lst = (" ".join(meta)).split(" ")
        for k, ent in enumerate(lst):
            if skip == 0:
                if ent == 'Energy':
                    meta_energy = lst[k + 1]
                    skip = 1
                elif ent == 'Method':
                    meta_method = lst[k + 1]
                    skip = 1
                elif ent == 'Type':
                    meta_type = lst[k + 1]
                    skip = 1
                elif ent == 'Name':
                    meta_name = lst[k + 1]
                    skip = 1
                elif ent == 'Path':
                    meta_path = lst[k + 1]
                    skip = 1
            # else:
            # sys.exit('parse_meta: Unknown keyword')
            else:
                skip = 0
    else:
        #print('building meta:')
        meta_energy = meta[-1]
        meta_force = meta[-2]
        meta_method = meta[2]
        try:
            meta_type = meta[3][0][1] + '_' + meta[3][1][1]
        except:
            meta_type = ''
        meta_xyz = [i[1] for i in meta[3] if i[0] == 'label'][0]
        meta_name = meta[1]
        #print("in parse_meta: meta:",meta)
        #print("in parse_meta: meta_energy:",meta_energy)
        #print("in parse_meta: meta_force:",meta_force)

    return meta_energy, meta_name, meta_method, meta_type, meta_name, meta_path, meta_xyz, meta_force


# ====================================================================================================

def sample_xid(meta_db, nsamp, npath=1, uniformdist=True):
    nblk = len(meta_db)
    if uniformdist:
        # extract unique names
        nm_uniq = np.unique(np.array([meta[1] for meta in meta_db]))
        # xyzid list of db for each name
        # Since the xyzid is unique within a given molecule configuration (e.g., CnHm),
        # the xyzid list store the molecule configuration and xyzid.
        rdxid = []
        for i in range(nm_uniq.shape[0]):
            list_nm = [[label[1].split('/') for label in meta[3] if label[0] == 'label'][0] for meta in meta_db if
                       meta[1] == nm_uniq[i]]
            if 'irc' in nm_uniq[i]:
                if len(list_nm) < npath:
                    print('for {} \nrequired points: {}\navailable points: {}'.format(nm_uniq[i], npath, len(list_nm)))
                    # sys.exit()
                else:
                    rdxid.extend(random.sample(list_nm, npath))
            else:
                if len(list_nm) < nsamp:
                    print('for {} \nrequired points: {}\navailable points: {}'.format(nm_uniq[i], npath, len(list_nm)))
                    # sys.exit()
                else:
                    rdxid.extend(random.sample(list_nm, nsamp))
        # sample random points from db for each name
    else:
        # sample random points from whole db
        if nblk < nsamp:
            print('required points: {}\navailable points: {}'.format(npath, nblk))
            # sys.exit()
        else:
            rdidx = random.sample(range(nblk), nsamp)
            try:
                rdxid = [[[label[1].split('/') for label in meta_db[i][3] if label[0] == 'label'][0]][0] for i in rdidx]
            except:
                rdxid = [meta_db[i][6].split('/') for i in rdidx]

    return rdxid



class Data_pes():
    """
	Data class relevant for PES construction and operations
	"""

    def __init__(self, atom_types=None):
        """
		Constructor for Data_pes class
		Defines atom_types and number of NNs
		"""
        if (atom_types):
            self.atom_types = atom_types
            self.num_nn = len(atom_types)

    def initialize(self, atom_types):
        self.atom_types = atom_types
        self.num_nn = len(atom_types)

    def tocuda(self, wloss=False, floss=False):
        self.xbtr = [[xt.to(self.device) for xt in self.xbtr[b]] for b in range(self.nbttr)]
        self.indtr = [[xt.to(self.device) for xt in self.indtr[b]] for b in range(self.nbttr)]
        self.ybtr = [self.ybtr[b].to(self.device) for b in range(self.nbttr)]
        if wloss:
            self.wbtr = [self.wbtr[b].to(self.device) for b in range(self.nbttr)]
        if floss:
            #self.dxbtr = [[torch.tensor(xt).to(self.device) for xt in self.dxbtr[b]] for b in range(self.nbttr)]
            self.fbtr = [torch.tensor(self.fbtr[b]).to(self.device) for b in range(self.nbttr)]
        else:
            print('There is no AEV derivative and/or force data.')
        try:
            self.xbts = [[xt.to(self.device) for xt in self.xbts[b]] for b in range(self.nbtts)]
            self.indts = [[xt.to(self.device) for xt in self.indts[b]] for b in range(self.nbtts)]
            self.ybts = [self.ybts[b].to(self.device) for b in range(self.nbtts)]
            if wloss:
                self.wbts = [self.wbts[b].to(self.device) for b in range(self.nbtts)]
            if floss:
                self.dxbts = [[xt.to(self.device) for xt in self.dxbts[b]] for b in range(self.nbtts)]
                self.fbts = [self.fbts[b].to(self.device) for b in range(self.nbtts)]
        except:
            print('There is no test set.')
        try:
            self.xbvl = [[xt.to(self.device) for xt in self.xbvl[b]] for b in range(self.nbtvl)]
            self.indvl = [[xt.to(self.device) for xt in self.indvl[b]] for b in range(self.nbtvl)]
            self.ybvl = [self.ybvl[b].to(self.device) for b in range(self.nbtvl)]
            if wloss:
                self.wbvl = [self.wbvl[b].to(self.device) for b in range(self.nbtvl)]
            if floss:
                self.dxbvl = [[xt.to(self.device) for xt in self.dxbvl[b]] for b in range(self.nbtvl)]
                self.fbvl = [self.fbvl[b].to(self.device) for b in range(self.nbtvl)]
        except:
            print('There is no validation set.')

        try:
            self.xbtr_lf = [[xt.to(self.device) for xt in self.xbtr_lf[b]] for b in range(self.nbttr_lf)]
            self.indtr_lf = [[xt.to(self.device) for xt in self.indtr_lf[b]] for b in range(self.nbttr_lf)]
            self.ybtr_lf = [self.ybtr_lf[b].to(self.device) for b in range(self.nbttr_lf)]
            if wloss:
                self.wbtr_lf = [self.wbtr_lf[b].to(self.device) for b in range(self.nbttr_lf)]
            if floss:
                self.dxbtr_lf = [[xt.to(self.device) for xt in self.dxbtr_lf[b]] for b in range(self.nbttr_lf)]
                self.fbtr_lf = [self.fbtr_lf[b].to(self.device) for b in range(self.nbttr_lf)]

            else:
                print('There is no AEV derivative and/or force data.')
            try:
                self.xbts_lf = [[xt.to(self.device) for xt in self.xbts_lf[b]] for b in range(self.nbtts_lf)]
                self.indts_lf = [[xt.to(self.device) for xt in self.indts_lf[b]] for b in range(self.nbtts_lf)]
                self.ybts_lf = [self.ybts_lf[b].to(self.device) for b in range(self.nbtts_lf)]
                if wloss:
                    self.wbts_lf = [self.wbts_lf[b].to(self.device) for b in range(self.nbtts_lf)]
                if floss:
                    self.dxbts_lf = [[xt.to(self.device) for xt in self.dxbts_lf[b]] for b in range(self.nbtts_lf)]
                    self.fbts_lf = [self.fbts_lf[b].to(self.device) for b in range(self.nbtts_lf)]
            except:
                print('There is no test set.')
            try:
                self.xbvl_lf = [[xt.to(self.device) for xt in self.xbvl_lf[b]] for b in range(self.nbtvl_lf)]
                self.indvl_lf = [[xt.to(self.device) for xt in self.indvl_lf[b]] for b in range(self.nbtvl_lf)]

                self.ybvl_lf = [self.ybvl_lf[b].to(self.device) for b in range(self.nbtvl_lf)]
                if wloss:
                    self.wbvl_lf = [self.wbvl_lf[b].to(self.device) for b in range(self.nbtvl_lf)]
                if floss:
                    self.dxbvl_lf = [[xt.to(self.device) for xt in self.dxbvl_lf[b]] for b in range(self.nbtvl_lf)]
                    self.fbvl_lf = [self.fbvl_lf[b].to(self.device) for b in range(self.nbtvl_lf)]
            except:
                print('There is no validation set.')
        except:
            print('There is no multifidelity dataset')
        print('input data was moved to cuda')


    # ==============================================================================================
    def prep_aev(self, atom_types=['C', 'H'], nrho_rad=32, nrho_ang=8, nalpha=8, R_c=[4.6, 3.1], beta=0.95):

        # define list of atom types in system
        # atom_types = ['C', 'H']

        # set values for radial and angular symmetry function parameters
        # nrho_rad = 32  # number of radial shells in the radial AEV
        # nrho_ang = 8  # number of radial shells in the angular AEV
        # nalpha = 8  # number of angular wedges dividing [0,pi] in the angular AEV

        # instantiate AEV for given atom types, for given
        try:
            import aevmod
            myaev = aevmod.aev(atom_types, nrho_rad, nrho_ang, nalpha, R_c, beta=beta)
        except:
            myaev = aev.Aev(atom_types, nrho_rad, nrho_ang, nalpha, R_c)
            # set the dimension of the data vector being the input for each NN in the system
            self.dimdat = myaev.dout
            print("Constructed aev, output dimensionality is:", myaev.dout)

        # init data class with list of atom types in system
        self.initialize(atom_types)

        return myaev

    # ==============================================================================================
    def aev_from_xyz(self, xyz_db, nrho_rad=32, nrho_ang=8, nalpha=8, R_c=[4.6, 3.1], pack_n_write=True, myaev=None, nblk=None, meta_db=None, beta=0.95):
        if myaev == None:

            # define list of atom types in system
            atom_types = ['C', 'H']

            # set values for radial and angular symmetry function parameters
            # nrho_rad = 32  # number of radial shells in the radial AEV
            # nrho_ang = 8  # number of radial shells in the angular AEV
            # nalpha = 8  # number of angular wedges dividing [0,pi] in the angular AEV

            # instantiate AEV for given atom types, for given
            try:
                import aevmod
                myaev = aevmod.aev(atom_types, nrho_rad, nrho_ang, nalpha, R_c, beta=beta)
            except:
                myaev = aev.Aev(atom_types, nrho_rad, nrho_ang, nalpha, R_c)
                # set the dimension of the data vector being the input for each NN in the system
                self.dimdat = myaev.dout
                print("Constructed aev, output dimensionality is:", myaev.dout)

            # init data class with list of atom types in system
            self.initialize(atom_types)

        else:
            atom_types = self.atom_types

        if nblk is None:
            nblk = len(xyz_db)
        if meta_db is None:
            parsed = [[0.0] for i in range(nblk)]
        else:
            parsed = parse_meta_db(meta_db)
        tag = None
        con = None

        # build daev database for available xyz database
        self.xyz_to_aev_db(xyz_db, nblk, parsed, myaev, tag, con, verbose=False, force=False)

        try:
            self.xyz_to_daev_db(xyz_db, nblk, myaev)
        except:
            pass

        return myaev


    # ==============================================================================================
    def get_data(self, args, xid=None):   #, testset=False):

        myaev = None

        if args.input_data_type == 'aev':
            # read AEV data base hdf5 file
            print("Reading aev hdf5 data base (currently implements one input file only)")

            hd.read_aev_db_hdf(self, args.input_data_fname[0], args.ni, args.nf)
            hd.unpack_data(self, fdata=args.floss)

            print("done reading aev db ...")

        elif args.input_data_type == 'pca':

            # read AEV PCA data base hdf5 file
            print("Reading aev pca hdf5 data base (currently implements one input file only)")

            hd.read_pca_aev_db_hdf(self, args.input_data_fname[0], args.ni, args.nf)
            hd.unpack_data(self)

            print("done reading aev pca db ...")

        else:

            # define list of atom types in system
            atom_types = ['C', 'H']

            # set values for radial and angular symmetry function parameters 
            nrho_rad = args.aev_params[0]  # number of radial shells in the radial AEV
            nrho_ang = args.aev_params[1]  # number of radial shells in the angular AEV
            nalpha = args.aev_params[2]    # number of angular wedges dividing [0,pi] in the angular AEV
            R_c = args.cuttoff_radius
            # instantiate AEV for given atom types, for given
            try:
                import aevmod
                myaev = aevmod.aev(atom_types, nrho_rad, nrho_ang, nalpha, R_c, beta=args.beta)
                print('Generating aevs using aevmod')
            except:
                myaev = aev.Aev(atom_types, nrho_rad, nrho_ang, nalpha, R_c)
                # set the dimension of the data vector being the input for each NN in the system
                self.dimdat = myaev.dout
                print('Generating aevs using aev.py')
                print("Constructed aev, output dimensionality is:", myaev.dout)

            # init data class with list of atom types in system
            self.initialize(atom_types)

            if args.input_data_type == 'xyz':
                print("parsing xyz data base (currently implements one input file only)")
                xyz_db, nblk, meta_db = db_parse_xyz(args.input_data_fname[0])
                parsed = parse_meta_db(meta_db)
                tag = None  # tag = 'b3lyp/6-31G'
                con = None  # con = xyz_db[0][0]

            elif args.input_data_type == 'sqlite':
                # during training set preparation, if xyzids for testset is specified,
                # read the xyzids, and exclude the xyzids when reading the sqlite db

                # read SQLite data base file 
                names = glob.glob(args.input_data_fname[0])
                print('input data files: ', names)
                if len(names) > 1:
                    print("parsing multiple SQLite xyz databases")
                    xyz_db, nblk, meta_db = multiple_sqldb_parse_xyz(names, fid=args.fidlevel, nameset=args.nameset,
                                                                     xid=xid, sort_ids=args.delta)
                    print('SQLite xyz data was extracted, fid: {}'.format(args.fidlevel))

                else:
                    print("parsing SQLite xyz data base", args.input_data_fname[0])
                    try:
                        if '/' in xid[0]:
                            xid_sep = [molid.split('/')[1] for molid in xid]
                        else:
                            xid_sep = xid
                    except:
                        xid_sep = xid
                    xyz_db, nblk, meta_db = sqldb_parse_xyz(names[0], fid=args.fidlevel,
                                                            nameset=args.nameset, xid=xid_sep, sort_ids=args.delta)
                
                parsed = parse_meta_db(meta_db)
                tag = None  # tag = 'b3lyp/6-31G'
                con = None  # con = xyz_db[0][0]
                force = args.wrf
                weights = args.wrw

            # build aev database for available xyz database
            print("Building PES AEV data base")
            self.xyz_to_aev_db(xyz_db, nblk, parsed, myaev, tag, con, force=force, weights=weights, verbose=False)

            if force:
                try:
                    print('Building derivative of AEV data base')
                    self.xyz_to_daev_db(xyz_db, nblk, myaev)
                except:
                    print('AEV derivatives were not calculated. Please check if aevmod is available.')

            # set the dimension of the data vector being the input for each NN in the system
            print("Constructed aev, output dimensionality is:", self.dimdat)

            if args.write_hdf:
                print("Packing data for writing aev_db_new.hdf5")
                hd.pack_data(self)

                print("Writing aev_db_new.hdf5")
                hd.write_aev_db_hdf(self, "aev_db_new.hdf5")
        print("done...")
        print("ndat:", self.ndat)
        print("dimdat:", self.dimdat)


        return myaev

    # ==========================================================================================


    def read_aev_db_txt(self, fname, myaev):
        """
		Method to read the AEV data base for the Data_pes object from a file
		"""
        # ==========================================================================================
        # Got ndat data points
        # each data point xdat[i] contains num_nn+1 objects, each of which is a list
        # xdat[i][0]        = [ [], [], ... [] ]   is a list of n0 lists
        #      n0 is the number of atoms of type 0 in the configuration
        #  and each inner list [] is the aev contents centered on the corresponding atom
        # xdat[i][1]        = [ [], [], ... [] ]   is a list of n1 lists
        #      n1 is the number of atoms of type 1 in the configuration
        #  and each inner list [] is the aev contents centered on the corresponding atom
        # ...
        # xdat[i][num_nn-1] = [ [], [], ... [] ]   is a list of n<num_nn-1> lists
        #      n<num_nn-1> is the number of atoms of type num_nn-1 in the configuration
        #  and each inner list [] is the aev contents centered on the corresponding atom
        # xdat[i][num_nn]   = []                   is a list containing 1 float
        #      this being the value of the energy for this data point
        #
        # tvtmsk[i] is a training-validation-testing mask, which will be defined as follows:
        # 	  1 if the data point will be part of the training   set    (default)
        #     0 if the data point will be part of the validation set (for hyper param optim)
        #    -1 if the data point will be part of the test       set
        # ntrdat is the number of training   points -- default ndat
        # nvldat is the number of validation points -- default 0
        # ntsdat is the number of test       points -- default 0
        # ==========================================================================================

        print("read_aev_db_txt: reading file:", fname)
        try:
            f = open(fname, "r")
        except IOError:
            print("Could not open file:" + fname)
        # sys.exit()
        with f:
            db = f.readlines()

        len_myaev = myaev.dout
        nline = len(db)
        k = 0
        got_types = list(db[k].strip('\n').split(" "))
        assert (checkEqual(self.atom_types, got_types))
        k += 1
        self.ndat = int(db[k])
        k += 1
        self.xdat = np.empty([self.ndat, self.num_nn + 1], dtype=object)
        self.full_symb_data = [[] for i in range(self.ndat)]
        for i in range(self.ndat):
            assert (int(db[k]) == self.num_nn + 1)
            k += 1
            for j in range(self.num_nn):
                numat = int(db[k])
                k += 1
                len_aev = int(db[k])
                assert (len_aev == len_myaev)
                k += 1
                dum = [list(db[k + 2 * q].strip('\n').split(" ")) for q in range(numat)]
                self.xdat[i][j] = [[float(p) for p in dm] for dm in dum]
                k += 2 * numat - 1
            k += 2
            self.xdat[i][self.num_nn] = [float(db[k])]
            k += 1
            self.full_symb_data[i] = list(db[k].strip('\n').split(" "))
            k += 1

        self.tvtmsk = np.ones([self.ndat], dtype=int)
        self.ntrdat = self.ndat
        self.nvldat = 0
        self.ntsdat = 0

        return

    def balanced_sampling_aev_db(self, ebin_width):
        # given
        # self.xdat
        # self.full_symb_data
        # self.pdat
        # self.meta

        # exclude data points at energies of more than <en_range> above the min-in-dataset for each class of config
        en_range = 50.0

        # potential energies
        en = [pes.kcpm(x[-1][0]) for x in self.xdat]

        # form a tuple from full_symb_data
        symbtup = [tuple(lst) for lst in self.full_symb_data]

        # find symb, the set of unique configs in the data
        symb = set(symbtup)
        N = len(symb)

        # build the list of index-lists isymb
        # if there are N configs in symb, then, for k=0,1,...,N-1
        # isymb[k] is the list of indices (indexing the data arrays xdat, etc above) indexing data samples for config k
        isymb = [[i for i in range(self.ndat) if symbtup[i] == s] for s in symb]

        for s, isl in zip(symb, isymb):
            print(''.join(s), "num:", len(isl))

        # en_min[k] is the min energy for config k in data
        # en_max[k] is the max energy for config k in data
        en_min = [min([en[i] for i in isl]) for isl in isymb]
        en_max = [max([en[i] for i in isl]) for isl in isymb]
        print("en_min:", en_min)
        print("en_max:", en_max)

        # iesymb[k] is the list of integers indexing data samples with energies in-range for config k
        iesymb = [[i for i in isl if en[i] < en_min[k] + en_range] for k, isl in enumerate(isymb)]

        # nc[k] is the # of config k in the data
        # ne[k] is the # of config k in the data with energies in the min+en_range zone
        nc = [len(isl) for isl in isymb]
        ne = [len(iel) for iel in iesymb]

        # en_max_in_range[k] is the max energy in-range for config k in data
        en_max_in_range = [max([en[i] for i in iel]) for iel in iesymb]

        print("en_max-en_min:", [em - en for em, en in zip(en_max, en_min)])
        print("en_max_in_range-en_min:", [em - en for em, en in zip(en_max_in_range, en_min)])
        print("N:", N, "\nnc:", nc, "\nne:", ne)

        nbin = [int((em - en) / ebin_width) + 1 for em, en in zip(en_max_in_range, en_min)]
        print([nbin[k] * ebin_width for k in range(N)])

        # iebsymb[k][b] is the list of integers indexing data samples with energies in bin b for config k
        iebsymb = [
            [[i for i in iel if en[i] >= en_min[k] + b * ebin_width and en[i] < en_min[k] + (b + 1) * ebin_width] for b
             in range(nbin[k])] for k, iel in enumerate(iesymb)]

        # mc[k][b] is the number of data samples of config k in energy bin b
        mc = [[len(iebsymb[k][b]) for b in range(nbin[k])] for k in range(N)]
        print("mc:", mc)

        # nthb[k] is # of bins that have number of samples >= bthr in config k
        bthr = 1000
        nthb = [sum([1 for b in mc[k] if b >= bthr]) for k in range(N)]
        print("nthb:", nthb)

        # mcmn is the minimum # of samples in a bin, among all bins containing more than bthr samples, among all config classes
        mcmn = min([min(np.array(mck)[np.array(mck) >= bthr]) for mck in mc])
        print("mcmn:", mcmn)

        print("total # of considered-bins among all configs:", sum(nthb))

        """
		# subdivide unity probability mass among all bins with more than bthr samples
		# thus each bin gets (1/sum_k=0^{N-1} nthb_k)
		# nq is the number of non-zero q[] 
		# init probabilities to zero for all data points
		qq = 1.0/float(sum(nthb))
		nq = 0
		q  = [0.0]*self.ndat
		for k in range(N):
			for b in range(nbin[k]):
				if mc[k][b] >= bthr:
					nq += mc[k][b]
					qqi = qq / float(mc[k][b])
					for i in iebsymb[k][b]:
						q[i] = qqi

		# check sum
		print("Probability sum:",sum(q))

		# identify by config,
		# consider there N different configs in the data
		# find # of data samples for each config ne[k], for k=0, 1, ..., N
		# this does not include samples that would be outside the sampled energy range
		# then set the probability of sampling from each config case k as P[k]=1/N

		# then employ binning over the range of available energies for each config k, by energy_bin_width (kcal/mol)
		# let there be nbin[k] bins for each config c[k]
		# let there be mc[k][j] data samples for config c[k] in bin b, where b=0,1,...,nbin[k]-1
		# and let the number of [>=bthr] bins for eack k be nthb[k], so that the number of [>=bthr] bins among all configs is sum(nthb)
		# then let the probability assigned to each data point of config k in bin b be (1/sum(nthb)) / mc[k][b]
		# put these in q[i] as the probability assigned to each data point i 

		# then use the numpy function numpy.random.choice()
		# e.g. np.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
		# to generate samples from the full data vector (all configs all energies) with the assigned element probabilities
		# we have n (=self.ndat) data points in total, all configs all energies
		# each data point is identified by the index i = 0, 1, ..., n
		# Then let's say we want to generate m samples (will be dictated by the min # samples per energy bin among all configs)
		# without replacement
		# we do this with 
		# ipt = np.random.choice(n,m,replace=False,q)
		# giving ipt as a m-long 1d array of integer indices pointing to i locations in the data array
		# we then use only these m samples from the data set ingoring the rest
		# these m-samples would be sampled uniformly among the configs in the data, 
		# and uniformly across the energy bins for each config
		# see: 
		# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.choice.html
		# https://stackoverflow.com/questions/45928993/sample-to-create-uniform-distribution-from-non-uniform-data
		# https://stackoverflow.com/questions/34077204/is-there-a-python-equivalent-to-rs-sample-function

		nsamp = mcmn * sum(nthb) 
		isthr = np.random.choice(self.ndat,nsamp,replace=False,p=q)
		print("Got",len(isthr)," balanced samples")
		"""

        # isthr is the list of indices of data points satisfying requirements
        # there are mcmn points in each bin, for each config k
        # these are sampled uniformly among the larger set of data points in each bin, without replacement
        # thus we have data that is uniformly sampled among valid energy bins and configs
        # first gen isthr as a list of lists, each of the latter containing sublists being lists of indices
        isthr = [[np.random.choice(iebsymb[k][b], mcmn, replace=False).tolist() for b, mck in enumerate(mc[k]) if
                  mck >= bthr] for k in range(N)]
        # flatten this 3-level nested listing to a 1-d list
        isthr = [i for sub in isthr for subsub in sub for i in subsub]

        print("Got", len(isthr), " balanced samples")

        self.ndat = len(isthr)
        self.full_symb_data = [self.full_symb_data[i] for i in isthr]
        self.xdat = [self.xdat[i] for i in isthr]
        self.pdat = [self.pdat[i] for i in isthr]
        self.meta = [self.meta[i] for i in isthr]
        self.tvtmsk = np.ones([self.ndat], dtype=int)
        self.ntrdat = self.ndat
        self.nvldat = 0
        self.ntsdat = 0

        # check
        """
		en = [pes.kcpm(x[-1][0]) for x in self.xdat]
		symbtup = [tuple(lst) for lst in self.full_symb_data]
		isymb   = [[i for i in range(self.ndat) if symbtup[i] == s] for s in symb]
		for s,isl in zip(symb,isymb):
			print(''.join(s),"num:", len(isl))
		en_min = [min([en[i] for i in isl]) for isl in isymb]
		en_max = [max([en[i] for i in isl]) for isl in isymb]
		print("en_min:",en_min)
		print("en_max:",en_max)

		iesymb = [[i for i in isl if en[i] < en_min[k]+en_range] for k,isl in enumerate(isymb)]
		nc = [len(isl) for isl in isymb]
		ne = [len(iel) for iel in iesymb]
		en_max_in_range = [max([en[i] for i in iel]) for iel in iesymb]
		print("en_max-en_min:",[em-en for em,en in zip(en_max,en_min)])
		print("en_max_in_range-en_min:",[em-en for em,en in zip(en_max_in_range,en_min)])
		print("N:",N,"\nnc:",nc,"\nne:",ne)

		nbin = [int((em-en)/ebin_width)+1 for em,en in zip(en_max_in_range,en_min)]
		print([nbin[k]*ebin_width for k in range(N)])
		"""

        return

    def random_shuffle_aev_db(self):
        try:
            mapIndexPosition = list(zip(self.xdat, self.full_symb_data, self.pdat, self.meta, self.padded_fdat, self.w, self.dxdat, self.fd2, self.fdat, self.padded_dxdat))
        except:
            try:
                mapIndexPosition = list(zip(self.xdat, self.full_symb_data, self.pdat, self.meta, self.padded_fdat, self.dxdat, self.fd2, self.fdat, self.padded_dxdat))
                print('Shuffling without w')
            except:
                mapIndexPosition = list(zip(self.xdat, self.full_symb_data, self.pdat, self.meta))
                print('Shuffling without forces')
        random.shuffle(mapIndexPosition)
        try:
            self.xdat, self.full_symb_data, self.pdat, self.meta, self.padded_fdat, self.w, self.dxdat, self.fd2, self.fdat, self.padded_dxdat = zip(*mapIndexPosition)
            print('random shuffle AEV, derivative of AEV, and weights')
        except:
            try:
                self.xdat, self.full_symb_data, self.pdat, self.meta, self.padded_fdat, self.dxdat, self.fd2, self.fdat, self.padded_dxdat = zip(*mapIndexPosition)
                print('Did some Shufflin')
            except:
                print('No Forces shuffle')
                self.xdat, self.full_symb_data, self.pdat, self.meta = zip(*mapIndexPosition)
        #print('self.meta: ', self.meta)

    def truncate_data(self, new_ndat):
        if self.trid_fname == None:
            self.ndat = new_ndat
            self.full_symb_data = self.full_symb_data[0:self.ndat]
            self.xdat = self.xdat[0:self.ndat]
            self.pdat = self.pdat[0:self.ndat]
            self.meta = self.meta[0:self.ndat]
            try:
                self.dxdat = self.dxdat[0:self.ndat]
                self.fdat = self.fdat[0:self.ndat]
                self.w = self.w[0:self.ndat]
                print('Truncate AEV, derivative of AEV, and weights')
            except:
                pass
            self.tvtmsk = self.tvtmsk[0:self.ndat]
            # self.ntrdat = self.ndat
            # self.nvldat = 0
            # self.ntsdat = 0
        else:
            print('dataset is truncated based on ', self.trid_fname)
            dat = np.loadtxt(self.trid_fname)
            ids = np.array(dat[:, 0], dtype=int)
            idall = np.array([int(self.meta[i][-2]) for i in range(self.ndat)])
            _, idall_id, dat_id = np.intersect1d(idall, ids, return_indices=True)
            newid = idall_id[0:new_ndat]
            # print(np.allclose(idall[newid], ids[dat_id]))
            self.ndat = new_ndat
            self.full_symb_data = np.array(self.full_symb_data)[newid].tolist()
            self.xdat = np.array(self.xdat)[newid].tolist()
            self.pdat = np.array(self.pdat)[newid].tolist()
            self.meta = np.array(self.meta)[newid].tolist()
            try:
                self.dxdat = np.array(self.dxdat)[newid].tolist()
                self.fdat = np.array(self.fdat)[newid].tolist()
                self.w = np.array(self.w)[newid].tolist()
                print('Truncate AEV, derivative of AEV, and weights')
            except:
                pass
            #
            # tvtmsk = np.ones(new_ndat, dtype=int)
            # tvtmsk[self.ntrdat:(self.ntrdat+self.nvldat)] = 0
            # tvtmsk[(self.ntrdat+self.nvldat):new_ndat] = -1
            # self.tvtmsk = tvtmsk.tolist()

    def write_aev_db_pickle(self, fname, myaev):
        d = {
            'laev': myaev.dout,
            'type': self.atom_types,
            'symb': self.full_symb_data,
            'xdat': self.xdat,
            'pdat': self.pdat,
            'meta': self.meta
        }
        # Pickle the 'data' dictionary using the highest protocol available.
        with open(fname, 'wb') as f:
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

    def read_aev_db_pickle(self, fname, myaev):
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        with open(fname, 'rb') as f:
            d = pickle.load(f)
            assert (d['laev'] == myaev.dout)
            assert (checkEqual(d['type'], self.atom_types))
            self.full_symb_data = d['symb']
            self.xdat = d['xdat']
            self.pdat = d['pdat']
            self.meta = d['meta']

        self.ndat = len(self.xdat)
        self.tvtmsk = np.ones([self.ndat], dtype=int)
        self.ntrdat = self.ndat
        self.nvldat = 0
        self.ntsdat = 0

    def write_aev_db_txt(self, fname):
        """
		Method to save the AEV data base for the Data_pes object to a file
		"""
        print("write_aev_db_txt: writing file:", fname)

        with open(fname, 'w') as f:
            print(*self.atom_types, file=f)
            print(self.ndat, file=f)
            for i in range(self.ndat):
                print(len(self.xdat[i]), file=f)
                for x in self.xdat[i]:
                    print(len(x), file=f)
                    for ax in x:
                        if type(ax) is list:
                            print(len(ax), file=f)
                            print(*ax, file=f)
                        else:
                            print("1", file=f)
                            print(ax, file=f)
                print(*self.full_symb_data[i], file=f)
        return



    def xyz_to_daev_db(self, xyz_db, nblk, myaev):
        """
        Method to build the derivative of AEV data base for the Data_pes object
        """
        #print("xyz_to_aev_db", verbose)
        aevmodule = myaev.__class__.__module__
        #print('aev module:', aevmodule)

        if aevmodule != 'aevmod':
            print('No aevmod')
            print('aevmod module requires for Jacobian calculation')
            sys.exit()

        J_tot = []
        idx = [0]
        self.full_symb_data_daev = []
        for blk in range(0, nblk):
            if blk == 0:
                prev_symb = xyz_db[blk][0]
            symb = xyz_db[blk][0]
            self.full_symb_data_daev.append(symb)
            if prev_symb != symb:
                idx.append(blk)
                prev_symb = symb

        idx.append(nblk)
        for i in range(idx.__len__() - 1):
            symb = xyz_db[idx[i]][0]
            conf = aevmod.config(symb)
            for j in range(idx[i], idx[i + 1]):
                if j == idx[i]:
                    x = np.array([xyz_db[j][1].flatten()])
                else:
                    x_new = np.array([xyz_db[j][1].flatten()])
                    x = np.concatenate((x, x_new))

            npt = conf.add_structures(x)
            myaev.build_index_sets(conf)
            J = np.array(myaev.eval_Jac(conf))

            # idx_C = [a == 'C' for a in symb]
            for k in range(0, npt):
                J_tot.append(J[k])


        self.ndat = nblk
        self.dxdat = np.empty([self.ndat, self.num_nn], dtype=object)

        for i in range(0, self.ndat):
            dxdatnn = [[]] * self.num_nn
            for j, s in enumerate(self.full_symb_data_daev[i]):
                k = self.atom_types.index(s)
                d = J_tot[i][j].tolist()
                
                dxdatnn[k] = dxdatnn[k] + [d]

            for k in range(0, self.num_nn):
                self.dxdat[i][k] = dxdatnn[k]
        
        #=====================================================
        d0 = len(self.dxdat)
        d1 = []
        d2 = []
        d3 = []
        d4 = []
        #print('padding dxdat')
        for i in range(len(self.dxdat)):
            d1.append(len(self.dxdat[i]))
            for j in range(len(self.dxdat[i])):
                d2.append(len(self.dxdat[i][j]))
                for k in range(len(self.dxdat[i][j])):
                    d3.append(len(self.dxdat[i][j][k]))
                    for v in range(len(self.dxdat[i][j][k])):
                        d4.append(len(self.dxdat[i][j][k][v]))


        self.padded_dxdat = np.zeros((d0, max(d1), max(d2), max(d3), max(d4)))
        for i in range(len(self.dxdat)):
            for j in range(len(self.dxdat[i])):
                for k in range(len(self.dxdat[i][j])):
                    for v in range(len(self.dxdat[i][j][k])):
                        self.padded_dxdat[i][j][k][v][:len(self.dxdat[i][j][k][v])] = self.dxdat[i][j][k][v]
        
        d0 = len(self.fdat)
        d1 = []
        d2 = []
        print('padding fdat!')
        for i in range(len(self.fdat)):
            d1.append(len(self.fdat[i]))
            for j in range(len(self.fdat[i])):
                d2.append(len(self.fdat[i][j]))
        self.fd2 = d2
        self.padded_fdat = np.zeros((d0, max(d1), max(d2)))
        for i in range(len(self.fdat)):
            for j in range(len(self.fdat[i])):
                self.padded_fdat[i][j][:len(self.fdat[i][j])] = self.fdat[i][j]

        #=====================================================
        return

    def xyz_to_aev_db(self, xyz_db, nblk, parsed, myaev, target_theory=None, target_symb=None, force=True, weights=None, verbose=False):
        """
		Method to build the AEV data base for the Data_pes object
		"""
        #print("xyz_to_aev_db:", target_theory, target_symb, verbose)
        #print('IN xyz_to_aev_db')
        aevmodule = myaev.__class__.__module__
        #print('xyz_to_aev_db: aev module:', aevmodule)

        full_aev_data = []
        full_energy_data = []
        if force:
            full_force_data = []
        self.full_symb_data = []
        self.pdat = []
        self.meta = []

        self.ndat = 0
        for blk in range(0, nblk):
            if target_theory:
                if parsed[blk][2] != target_theory:
                    continue
            if target_symb:
                if not checkEqual(xyz_db[blk][0], target_symb):
                    continue

            energy = parsed[blk][0]

            if aevmodule == 'aevmod':
                symb = xyz_db[blk][0]
                conf = aevmod.config(symb)
                x = np.array([xyz_db[blk][1].flatten()])
                npt = conf.add_structures(x)
                myaev.build_index_sets(conf)
                y = np.array(myaev.eval(conf)[0])
            elif aevmodule == 'aev':
                conf = aev.Config(xyz_db[blk][0], xyz_db[blk][1])
                symb = conf.get_chemical_symbols()
                x = np.array([conf.get_positions().flatten()])

                if verbose:
                    print("blk:", format(blk, '04d'), "ndat:", format(self.ndat, '04d'), energy,
                          '%s' % ''.join(
                              [t + str(symb.count(t)) if symb.count(t) > 1 else t if symb.count(t) > 0 else '' for t in
                               myaev.types]))

                conf.set_index_sets(*myaev.bld_index_sets(symb))
                y = myaev.eval(symb, *conf.get_index_sets(), x)[0]  # evaluate AEV

            full_aev_data.append(y)
            full_energy_data.append(energy)
            if force:
                full_force_data.append(parsed[blk][-1].flatten())
            self.full_symb_data.append(symb)
            self.ndat = self.ndat + 1
            self.pdat.append(np.reshape(x, (-1, 3)))
            self.meta.append(parsed[blk])


        self.dimdat = full_aev_data[0].shape[-1]


        # ==========================================================================================
        # Got ndat data points
        # each data point xdat[i] contains num_nn+1 objects, each of which is a list
        # xdat[i][0]        = [ [], [], ... [] ]   is a list of n0 lists
        #      n0 is the number of atoms of type 0 in the configuration
        #  and each inner list [] is the aev contents centered on the corresponding atom
        # xdat[i][1]        = [ [], [], ... [] ]   is a list of n1 lists
        #      n1 is the number of atoms of type 1 in the configuration
        #  and each inner list [] is the aev contents centered on the corresponding atom
        # ...
        # xdat[i][num_nn-1] = [ [], [], ... [] ]   is a list of n<num_nn-1> lists
        #      n<num_nn-1> is the number of atoms of type num_nn-1 in the configuration
        #  and each inner list [] is the aev contents centered on the corresponding atom
        # xdat[i][num_nn]   = []                   is a list containing 1 float
        #      this being the value of the energy for this data point
        #
        # tvtmsk[i] is a training-validation-testing mask, which will be defined as follows:
        # 	  1 if the data point will be part of the training   set    (default)
        #     0 if the data point will be part of the validation set (for hyper param optim)
        #    -1 if the data point will be part of the test       set
        # ntrdat is the number of training   points -- default ndat
        # nvldat is the number of validation points -- default 0
        # ntsdat is the number of test       points -- default 0
        # ==========================================================================================

        if weights == None:
            self.w = np.ones(self.ndat)
        else:
            if weights == 'z':
                nzone = 10
                zones = np.linspace(np.min(full_energy_data), np.max(full_energy_data), nzone)
                zones = zones[:-1]

                zid = np.digitize(full_energy_data, zones)
                zuniq, zcnts = np.unique(zid, return_counts=True)
                wt = 1 / zuniq.shape[0]

                self.w = np.empty(self.ndat)
                for i in range(0, self.ndat):
                    for j in range(zuniq.shape[0]):
                        if zid[i] == zuniq[j]:
                            self.w[i] = wt / zcnts[j]
            elif weights == 'c':
                names = []
                for i in range(self.ndat):
                    name = parsed[i][1]
                    names.append(name)
                zuniq, zcnts = np.unique(names, return_counts=True)
                wt = 1 / zuniq.shape[0]

                self.w = np.empty(self.ndat)
                for i in range(0, self.ndat):
                    for j in range(zuniq.shape[0]):
                        if names[i] == zuniq[j]:
                            self.w[i] = wt / zcnts[j]


        self.xdat = np.empty([self.ndat, self.num_nn + 1], dtype=object)
        if force:
            self.fdat = np.empty([self.ndat, 1], dtype='float64').tolist()

        for i in range(0, self.ndat):
            xdatnn = [[]] * self.num_nn
            for j, s in enumerate(self.full_symb_data[i]):
                k = self.atom_types.index(s)
                d = full_aev_data[i][j].tolist()
                xdatnn[k] = xdatnn[k] + [d]

            for k in range(0, self.num_nn):
                self.xdat[i][k] = xdatnn[k]

            self.xdat[i][self.num_nn] = [float(full_energy_data[i])]
            if force:
                self.fdat[i] = [full_force_data[i]]
        self.tvtmsk = np.ones([self.ndat], dtype=int)
        self.ntrdat = self.ndat
        self.nvldat = 0
        self.ntsdat = 0

        if verbose:
            with np.printoptions(precision=4, suppress=True):
                for i in range(self.ndat):
                    print(self.meta[i])
                    print(self.pdat[i])
                    print(self.full_symb_data[i])
                    for t in range(self.num_nn + 1):
                        print(np.array(self.xdat[i][t]))

        return

    def set_tvt_mask(self, tvtmsk=None, tvt=None):

        """
		Method to set masks defining the subsetting of the data into
		training, validation, and testing subsets
		tvtmsk[i] is ['molecule-name/xyzid', 'tag'], e.g.  ['C5H5/842314', '2'], with i=0,...,len(tvtmsk)-1
		where tag is:
		      -1 if the data point is for testing
		   and: 
		       2 .................... for training and validation
		   or: 
		       0 .................... for validation
		       1 .................... for training
        """

        #print(tvtmsk)
        #ltrvl = len([a[0] for a in tvtmsk if a[1] == '2'])
        #lts   = len([a[0] for a in tvtmsk if a[1] == '-1'])
        #lvl   = len([a[0] for a in tvtmsk if a[1] == '0'])
        #ltr   = len([a[0] for a in tvtmsk if a[1] == '1'])
        #print('in set_tvt_mask: tvtmsk[0]:',tvtmsk[0],', len:',len(tvtmsk),', tvt:',tvt)
        #print('ltrvl:',ltrvl)
        #print('lts  :',lts)
        #print('lvl  :',lvl)
        #print('ltr  :',ltr)
        

        if tvtmsk is not None and tvt is not None:
            if tvt[2] > 0:
                print('Error: Test set was set based on pre-set tvt mask file. The proportion of test set in the \'--tvt\' flag should be 0.')
                sys.exit()

        if tvtmsk is None:
            print('tvtmsk is None')
            assert (self.ntrdat >= 0 and self.nvldat >= 0 and self.ntsdat >= 0)
            assert (self.ntrdat + self.nvldat + self.ntsdat == self.ndat)

            # enforce the default
            self.tvtmsk = np.ones([self.ndat], dtype=int)

            nvt = self.nvldat + self.ntsdat
            if nvt == 0:
                return 0

            # generate nvt random *unique* integer samples in [0,ndat)
            # see: https://stackoverflow.com/questions/22842289/generate-n-unique-random-numbers-within-a-range
            if self.tvt_shuffle:
                ivt = random.sample(range(0, self.ndat), nvt)
                print('tvt mask was set randomly.')
            else:
                ivt = range(0, nvt)
            for i in ivt[0:self.nvldat]:
                self.tvtmsk[i] = 0

            for i in ivt[self.nvldat:nvt]:
                self.tvtmsk[i] = -1
            return 0
        else:
            #mla[i] is the 'molecule-name/xyzid' for data point i=0,..,ndat-1
            mla = np.array([m[-2] for m in self.meta])

            # enforce the default, 1->for training
            self.tvtmsk = np.ones([self.ndat], dtype=int)

            if tvt is None:
                tvt = [0.8, 0.2, 0.]
            if '2' in [a[1] for a in tvtmsk]:
                tvids = [a[0] for a in tvtmsk if a[1] == '2']
                print("tvids[0]:",tvids[0],' len:',len(tvids))
                #print(tvids)
                ntv = len(tvids)
            else:
                vlids = [a[0] for a in tvtmsk if a[1] == '0']
                print("vlids[0]:",vlids[0],' len:',len(vlids))
                trids = [a[0] for a in tvtmsk if a[1] == '1']
                print("trids[0]:",trids[0],' len:',len(trids))
                ntv = len(vlids) + len(trids)

            tstids = [a[0] for a in tvtmsk if a[1] == '-1']
            print("tstids[0]:",tstids[0],' len:',len(tstids))
            print("ntv:",ntv)

            # find and tag all the points in the dataset which are also tagged
            # in the tvt<>.txt input file as for testing
            for i in tstids:
                #ii = [idx for idx in range(self.ndat) if self.meta[idx][-2] == i][0]
                #self.tvtmsk[ii] = -1
                ii = np.where(mla == i)[0]
                if ii.size == 1:
                    self.tvtmsk[ii[0]] = -1
                elif ii.size > 1:
                    sys.exit('set_tvt_mask: found multiple matches in tstids')

            ntvt = sum(tvt)
            print('set_tvt_mask: ntvt,i.e. sum(tvt):',ntvt,', tvt:',tvt)

            if ntvt > 1:
                self.ntrdat = int(tvt[0])
                self.nvldat = int(tvt[1])
                self.ntsdat = len(tstids)
            else:
                self.ntrdat = int(tvt[0] * ntv)
                self.ntsdat = len(tstids)
                if ntvt == 1:
                    self.nvldat = self.ndat - self.ntrdat - self.ntsdat
                    if self.nvldat < 0:
                        print('data.py: set_tvt_mask: self.nvldat=',self.nvldat)
                        sys.exit('exiting in data.py')
                else:
                    self.nvldat = int(tvt[1] * ntv)
                    print('!!!WARNING!!! sum of tvt mask is not 1. You are using only partial dataset: {}%'.format(ntvt*100))

            print("self.ntrdat:",self.ntrdat)
            print("self.nvldat:",self.nvldat)
            print("self.ntsdat:",self.ntsdat)
            numtvt=self.ntrdat+self.nvldat+self.ntsdat
            print('numtvt:',numtvt,', self.ndat:',self.ndat)
            if numtvt > self.ndat:
                print('WARNING: based on specified tvt mask, numtvt > self.ndat')
                print('It will be over-ruled below just fyi')

            if self.nvldat > 0:
                if '2' in [a[1] for a in tvtmsk]:
                    if self.tvt_shuffle:
                        print(len(tvids), self.nvldat)
                        iv = random.sample(tvids, self.nvldat)
                        print('tvt mask for training/validation was set randomly.')
                    else:
                        iv = tvids[:self.nvldat]
                else:
                    iv = vlids

                if '/' not in self.meta[0][-2] and '/' in iv[0]:
                    iv = [i.split('/')[-1] for i in iv]

                # find and tag all the points in the dataset which are also tagged
                # in the tvt<>.txt input file as for validation
                for i in iv:
                    #ii = [idx for idx in range(self.ndat) if self.meta[idx][-2] == i][0]
                    #self.tvtmsk[ii] = 0
                    ii = np.where(mla == i)[0]
                    if ii.size == 1:
                        self.tvtmsk[ii[0]] = 0
                    elif ii.size > 1:
                        sys.exit('set_tvt_mask: found multiple matches in vlids')
            numts = len([i for i in range(0, self.ndat) if self.tvtmsk[i] == -1])
            numvl = len([i for i in range(0, self.ndat) if self.tvtmsk[i] ==  0])
            numtr = len([i for i in range(0, self.ndat) if self.tvtmsk[i] ==  1])
            numtv = len([i for i in range(0, self.ndat) if self.tvtmsk[i] ==  2])
            print('set_tvt_mask: final check on contents of self.tvtmsk')
            print('numts:',numts,', numvl:',numvl,', numtr:',numtr,', numtv:',numtv,', ndat:',self.ndat)
            if self.ntsdat != numts:
                sys.exit('mismatch in numts and self.ntsdat') 
            if self.nvldat != numvl:
                print(self.nvldat, numvl)
                sys.exit('mismatch in numvl and self.nvldat') 
            if self.ntrdat != numtr:
                print('mismatch in numtr and self.ntrdat') 

            print("self.ntrdat:",self.ntrdat)
            print("self.nvldat:",self.nvldat)
            print("self.ntsdat:",self.ntsdat)

            numtvt=self.ntrdat+self.nvldat+self.ntsdat
            print('numtvt:',numtvt,', self.ndat:',self.ndat)

            return 0

    def write_tvt_mask(self, fname):
        try:
            f = open(fname, "w")
        except IOError:
            print("write_tvt_mask: could not open file:", fname)
        # sys.exit()
        with f:
            print("write_tvt_mask: writing file: ... ", fname, end='')
            for i, msk in enumerate(self.tvtmsk):
                print('{:9d} {:3d}'.format(i, msk), file=f)
            print(" done")

    def prep_data(self, idl, num_batches=1, verbose=False):
        """
		Method to prepare data by packaging it appropriately for NN batch computations
		"""
        # tr[i] maps i=range(ntrdat) to range(ndat)
        # thus maps the 0,...,ntrdat-1 training points to the full array indices 0,...,ndat-1
        # this is useful for extracting the training data subset of xdat into xb,yb

        self.ndl = len(idl)

        self.nat = [[self.full_symb_data[idl[i]].count(self.atom_types[t]) for i in range(self.ndl)] for t in
                    range(self.num_nn)]

        self.nbt = num_batches
        bsz = int(self.ndl / self.nbt)

        self.bi = [b * bsz for b in range(self.nbt)]  # starting data index for each batch, in the *training data set*
        self.bf = [(b + 1) * bsz for b in range(self.nbt)]  # ending data index for each batch, in the training data set
        self.bf[self.nbt - 1] = self.ndl  # let the last batch end at ntrdat exactly


        self.xb = [[[] for j in range(self.num_nn)] for b in range(self.nbt)]
        self.yb = [[] for b in range(self.nbt)]
        self.wb = [[] for b in range(self.nbt)]
        for b in range(self.nbt):
            for j in range(self.num_nn):
                self.xb[b][j] = torch.tensor(
                    [self.xdat[idl[i]][j][k] for i in range(self.bi[b], self.bf[b]) for k in range(self.nat[j][i])])
            self.yb[b] = torch.tensor([self.xdat[idl[i]][self.num_nn] for i in range(self.bi[b], self.bf[b])])
            self.wb[b] = torch.tensor([self.w[idl[i]] for i in range(self.bi[b], self.bf[b])])

        try:
            self.dxb = [[[] for j in range(self.num_nn)] for b in range(self.nbt)]
            for b in range(self.nbt):
                for j in range(self.num_nn):
                    self.dxb[b][j] = [self.dxdat[idl[i]][j][k] for i in range(self.bi[b], self.bf[b]) for k in range(self.nat[j][i])]
        except:
            pass

        try:
            self.fb = [[] for b in range(self.nbt)]
            for b in range(self.nbt):
                for j in range(self.num_nn):
                    self.fb[b] = torch.tensor([self.fdat[idl[i]][0] for i in range(self.bi[b], self.bf[b])])
        except:
            pass



        self.kk = [[[sum(self.nat[t][self.bi[b]:i]) for i in range(self.bi[b], self.bf[b])] for t in range(self.num_nn)]
                   for b in range(self.nbt)]




        self.nab = [[[self.nat[t][i] for i in range(self.bi[b], self.bf[b])] for t in range(self.num_nn)] for b in
                    range(self.nbt)]

        self.inddl = [
            [torch.zeros(sum([self.nab[b][t][k] for k in range(self.bf[b] - self.bi[b])]), dtype=int) for t in
             range(self.num_nn)] for b in range(self.nbt)]
        for b in range(self.nbt):
            for t in range(self.num_nn):
                for k in range(self.bf[b] - self.bi[b]):
                    self.inddl[b][t][self.kk[b][t][k]:self.kk[b][t][k] + self.nab[b][t][k]] = k
        
        return 0

    def prep_training_data(self, num_batches=1, verbose=False, bpath=None):
        """
		Method to prepare training data by packaging it appropriately for NN batch computations
		"""
        # itr[i] maps i=range(ntrdat) to range(ndat)
        # thus maps the 0,...,ntrdat-1 training points to the full array indices 0,...,ndat-1
        # this is useful for extracting the training data subset of xdat into xb,yb
        # ni and nf provide further optional subsetting of the data, by selecting the ni:nf range of itr

        itr = [i for i in range(0, self.ndat) if self.tvtmsk[i] == 1]

        self.nattr = [[self.full_symb_data[itr[i]].count(self.atom_types[t]) for i in range(self.ntrdat)] for t in
                      range(self.num_nn)]
        
        nattr_maxs = [max(i) for i in self.nattr]
        print('nattr_maxs: ', nattr_maxs)
        if verbose:
            print("nat:", self.nattr)

        self.nbttr = num_batches
        bsz = int(math.ceil(self.ntrdat / self.nbttr))

        self.bitr = [b * bsz for b in
                     range(self.nbttr)]  # starting data index for each batch, in the *training data set*
        self.bftr = [(b + 1) * bsz for b in
                     range(self.nbttr)]  # ending data index for each batch, in the training data set
        self.bftr[self.nbttr - 1] = self.ntrdat  # let the last batch end at ntrdat exactly

        if verbose:
            print("bi:", self.bitr, "\nbf:", self.bftr)


        self.xbtr = [[[] for j in range(self.num_nn)] for b in range(self.nbttr)]
        self.ybtr = [[] for b in range(self.nbttr)]
        self.wbtr = [[] for b in range(self.nbttr)]
        for b in range(self.nbttr):
            for j in range(self.num_nn):
                self.xbtr[b][j] = torch.tensor(
                    [self.xdat[itr[i]][j][k] for i in range(self.bitr[b], self.bftr[b]) for k in
                     range(self.nattr[j][i])])
            self.ybtr[b] = torch.tensor([self.xdat[itr[i]][self.num_nn] for i in range(self.bitr[b], self.bftr[b])])
            try:
                self.wbtr[b] = torch.tensor([self.w[itr[i]] for i in range(self.bitr[b], self.bftr[b])])
            except:
                pass

        
        try:
            #self.dxbtr = [[[] for j in range(self.num_nn)] for b in range(self.nbttr)]
            for b in range(self.nbttr):
                dxtr = [[] for j in range(self.num_nn)]
                for j in range(self.num_nn):
                    #self.dxbtr[b][j] = torch.tensor([self.padded_dxdat[itr[i]][j][k] for i in range(self.bitr[b], self.bftr[b]) for k in range(nattr_maxs[j])])
                    dxtr[j] = torch.tensor([self.padded_dxdat[itr[i]][j][k] for i in range(self.bitr[b], self.bftr[b]) for k in range(nattr_maxs[j])])
                torch.save(dxtr, bpath+'dxtr_batch_'+str(b))


        
        except:
            pass

        try:
            self.fbtr = [[] for b in range(self.nbttr)]
            self.pfdims_tr = [[] for b in range(self.nbttr)]
            for b in range(self.nbttr):
                self.pfdims_tr[b].append([self.fd2[itr[i]] for i in range(self.bitr[b], self.bftr[b])])
                #self.fbtr[b] = torch.tensor([self.padded_fdat[itr[i]][0] for i in range(self.bitr[b], self.bftr[b])])
                ftr = torch.tensor([self.padded_fdat[itr[i]][0] for i in range(self.bitr[b], self.bftr[b])])
                torch.save(ftr, bpath+'ftr_batch_'+str(b))
            torch.save(self.pfdims_tr, bpath + 'pfims_tr')

        except:
            pass


        self.kktr = [[[sum(self.nattr[t][self.bitr[b]:i]) for i in range(self.bitr[b], self.bftr[b])] for t in
                      range(self.num_nn)] for b in range(self.nbttr)]


        self.nabtr = [[[self.nattr[t][i] for i in range(self.bitr[b], self.bftr[b])] for t in range(self.num_nn)] for b
                      in range(self.nbttr)]


        self.indtr = [
            [torch.zeros(sum([self.nabtr[b][t][k] for k in range(self.bftr[b] - self.bitr[b])]), dtype=int) for t in
             range(self.num_nn)] for b in range(self.nbttr)]
        for b in range(self.nbttr):
            for t in range(self.num_nn):
                for k in range(self.bftr[b] - self.bitr[b]):
                    self.indtr[b][t][self.kktr[b][t][k]:self.kktr[b][t][k] + self.nabtr[b][t][k]] = k


        return 0

    def prep_validation_data(self, num_batches=1, verbose=False, bpath = None):
        """
		Method to prepare validation data by packaging it appropriately for NN batch computations
		"""
        print('data.py: prep_validation_data')

        ivl = [i for i in range(0, self.ndat) if self.tvtmsk[i] == 0]

        print('ivl size:',len(ivl))
        print('ndat:',self.ndat,', nvldat:',self.nvldat,', num_nn:',self.num_nn)

        self.natvl = [[self.full_symb_data[ivl[i]].count(self.atom_types[t]) for i in range(self.nvldat)] for t in
                      range(self.num_nn)]
        if verbose:
            print("natvl:", self.natvl)

        self.nbtvl = num_batches
        bsz = int(self.nvldat / self.nbtvl)

        self.bivl = [b * bsz for b in
                     range(self.nbtvl)]  # starting data index for each batch, in the *training data set*
        self.bfvl = [(b + 1) * bsz for b in
                     range(self.nbtvl)]  # ending data index for each batch, in the training data set
        self.bfvl[self.nbtvl - 1] = self.nvldat  # let the last batch end at ntrdat exactly

        if verbose:
            print("bivl:", self.bivl, "\nbfvl:", self.bfvl)




        self.xbvl = [[[] for j in range(self.num_nn)] for b in range(self.nbtvl)]
        self.ybvl = [[] for b in range(self.nbtvl)]
        self.wbvl = [[] for b in range(self.nbtvl)]
        for b in range(self.nbtvl):
            for j in range(self.num_nn):
                self.xbvl[b][j] = torch.tensor(
                    [self.xdat[ivl[i]][j][k] for i in range(self.bivl[b], self.bfvl[b]) for k in
                     range(self.natvl[j][i])])
            self.ybvl[b] = torch.tensor([self.xdat[ivl[i]][self.num_nn] for i in range(self.bivl[b], self.bfvl[b])])
            try:
                self.wbvl[b] = torch.tensor([self.w[ivl[i]] for i in range(self.bivl[b], self.bfvl[b])])
            except:
                pass
        try:
            self.dxbvl = [[[] for j in range(self.num_nn)] for b in range(self.nbtvl)]
            for b in range(self.nbtvl):
                dxvl = [[] for j in range(self.num_nn)]
                for j in range(self.num_nn):
                    #self.dxbvl[b][j] = torch.tensor([self.padded_dxdat[ivl[i]][j][k] for i in range(self.bivl[b], self.bfvl[b]) for k in
                    #                  range(self.natvl[j][i])])
                    dxvl[j] = torch.tensor([self.padded_dxdat[ivl[i]][j][k] for i in range(self.bivl[b], self.bfvl[b]) for k in
                                      range(self.natvl[j][i])])
                    #self.dxbvl[b][j] = torch.tensor([self.dxdat[ivl[i]][j][k] for i in range(self.bivl[b], self.bfvl[b]) for k in
                    #                  range(self.natvl[j][i])])
                torch.save(dxvl, bpath+'dxvl_batch_'+str(b))
        except:
            pass

        try:
            self.fbvl = [[] for b in range(self.nbtvl)]
            self.pfdims_vl = [[] for b in range(self.nbtvl)]
            for b in range(self.nbtvl):
                self.pfdims_vl[b].append([self.fd2[ivl[i]] for i in range(self.bivl[b], self.bfvl[b])])
                #self.fbvl[b] = torch.tensor([self.padded_fdat[ivl[i]][0] for i in range(self.bivl[b], self.bfvl[b])])
                fvl = torch.tensor([self.padded_fdat[ivl[i]][0] for i in range(self.bivl[b], self.bfvl[b])])
                torch.save(fvl, bpath+'fvl_batch_'+str(b))
                #for j in range(self.num_nn):
                    #self.fbvl[b] = torch.tensor([self.padded_fdat[ivl[i]][0] for i in range(self.bivl[b], self.bfvl[b])])
                    #self.fbvl[b] = torch.tensor([self.fdat[ivl[i]][0] for i in range(self.bivl[b], self.bfvl[b])])
            torch.save(self.pfdims_vl, bpath + 'pfims_vl')
        except:
            pass


        self.kkvl = [[[sum(self.natvl[t][self.bivl[b]:i]) for i in range(self.bivl[b], self.bfvl[b])] for t in
                      range(self.num_nn)] for b in range(self.nbtvl)]


        self.nabvl = [[[self.natvl[t][i] for i in range(self.bivl[b], self.bfvl[b])] for t in range(self.num_nn)] for b
                      in range(self.nbtvl)]

        self.indvl = [
            [torch.zeros(sum([self.nabvl[b][t][k] for k in range(self.bfvl[b] - self.bivl[b])]), dtype=int) for t in
             range(self.num_nn)] for b in range(self.nbtvl)]
        for b in range(self.nbtvl):
            for t in range(self.num_nn):
                for k in range(self.bfvl[b] - self.bivl[b]):
                    self.indvl[b][t][self.kkvl[b][t][k]:self.kkvl[b][t][k] + self.nabvl[b][t][k]] = k


        return 0

    def prep_testing_data(self, num_batches=1, verbose=False, bpath=None):
        """
		Method to prepare testing data by packaging it appropriately for NN batch computations
		"""

        its = [i for i in range(0, self.ndat) if self.tvtmsk[i] == -1]

        self.natts = [[self.full_symb_data[its[i]].count(self.atom_types[t]) for i in range(self.ntsdat)] for t in
                      range(self.num_nn)]
        if verbose:
            print("natts:", self.natts)
        
        #num_batches = 20
        self.nbtts = num_batches
        bsz = int(self.ntsdat / self.nbtts)

        self.bits = [b * bsz for b in
                     range(self.nbtts)]  # starting data index for each batch, in the *training data set*
        self.bfts = [(b + 1) * bsz for b in
                     range(self.nbtts)]  # ending data index for each batch, in the training data set
        self.bfts[self.nbtts - 1] = self.ntsdat  # let the last batch end at ntrdat exactly

        if verbose:
            print("bits:", self.bits, "\nbfts:", self.bfts)
        
        self.xbts = [[[] for j in range(self.num_nn)] for b in range(self.nbtts)]
        self.ybts = [[] for b in range(self.nbtts)]
        self.labelbts = [[] for b in range(self.nbtts)]
        self.wbts = [[] for b in range(self.nbtts)]
        for b in range(self.nbtts):
            for j in range(self.num_nn):
                self.xbts[b][j] = torch.tensor(
                    [self.xdat[its[i]][j][k] for i in range(self.bits[b], self.bfts[b]) for k in
                     range(self.natts[j][i])])
            self.ybts[b] = torch.tensor([self.xdat[its[i]][self.num_nn] for i in range(self.bits[b], self.bfts[b])])
            self.labelbts[b] = [self.meta[its[i]][-2] for i in range(self.bits[b], self.bfts[b])]
            try:
                self.wbts[b] = torch.tensor([self.w[its[i]] for i in range(self.bits[b], self.bfts[b])])
            except:
                pass
        
        
        
        try:
            #self.dxbts = [[[] for j in range(self.num_nn)] for b in range(self.nbtts)]
            for b in range(self.nbtts):
                dxts = [[] for j in range(self.num_nn)]
                for j in range(self.num_nn):
                    #self.dxbts[b][j] = torch.tensor([self.padded_dxdat[its[i]][j][k] for i in range(self.bits[b], self.bfts[b]) for k in
                    #                  range(self.natts[j][i])])
                    dxts[j] = torch.tensor([self.padded_dxdat[its[i]][j][k] for i in range(self.bits[b], self.bfts[b]) for k in
                                      range(self.natts[j][i])])
                    #self.dxbts[b][j] = torch.tensor([self.dxdat[its[i]][j][k] for i in range(self.bits[b], self.bfts[b]) for k in
                    #                  range(self.natts[j][i])])
                torch.save(dxts, bpath+'dxts_batch_'+str(b))
        except:
            pass

        try:
            #self.fbts = [[] for b in range(self.nbtts)]
            self.pfdims_ts = [[] for b in range(self.nbtts)]
            for b in range(self.nbtts):
                self.pfdims_ts[b].append([self.fd2[its[i]] for i in range(self.bits[b], self.bfts[b])])
                #self.fbts[b] = torch.tensor([self.padded_fdat[its[i]][0] for i in range(self.bits[b], self.bfts[b])])
                fts = torch.tensor([self.padded_fdat[its[i]][0] for i in range(self.bits[b], self.bfts[b])])
                torch.save(fts, bpath+'fts_batch_'+str(b))
                #for j in range(self.num_nn):
                    #self.fbts[b] = torch.tensor([self.padded_fdat[its[i]][0] for i in range(self.bits[b], self.bfts[b])])
                    #self.fbts[b] = torch.tensor([self.fdat[its[i]][0] for i in range(self.bits[b], self.bfts[b])])
            torch.save(self.pfdims_ts, bpath + 'pfims_ts')
        except:
            pass


        self.kkts = [[[sum(self.natts[t][self.bits[b]:i]) for i in range(self.bits[b], self.bfts[b])] for t in
                      range(self.num_nn)] for b in range(self.nbtts)]



        self.nabts = [[[self.natts[t][i] for i in range(self.bits[b], self.bfts[b])] for t in range(self.num_nn)] for b
                      in range(self.nbtts)]

        self.indts = [
            [torch.zeros(sum([self.nabts[b][t][k] for k in range(self.bfts[b] - self.bits[b])]), dtype=int) for t in
             range(self.num_nn)] for b in range(self.nbtts)]
        for b in range(self.nbtts):
            for t in range(self.num_nn):
                for k in range(self.bfts[b] - self.bits[b]):
                    self.indts[b][t][self.kkts[b][t][k]:self.kkts[b][t][k] + self.nabts[b][t][k]] = k

        return 0


# ====================================================================================================
# check that two lists match exactly
def checkEqual(L1, L2):
    return len(L1) == len(L2) and L1 == L2


# ====================================================================================================
def db_parse_xyz(name):
    """ Reads xyz database file and returns a list of lists, each of the latter containing two items:
		1) a list of strings, being chemical symbols of each atom in the configuration
		   e.g. ["H","H","O"] for a configuration with 2 H and 1 O atoms
		2) a 2d numpy array with n_atom rows and 3 columns, where
		   n_atom is the number of atoms in the configuration, and where
		   each row is the (x,y,z) coordinates of an atom in the configuration
		This is intended to be a means to read a file that's a concatenatenation of many xyz files
		each block can be a different configuration (e.g. different molecule) or the same config but in
		a different geometry
		If a config contains an atom not in the list of types, we stop later
	"""
    try:
        f = open(name, "r")
    except IOError:
        print("Could not open file:" + name)
    # sys.exit()
    with f:
        xyz = f.readlines()

    n_line = len(xyz)

    blk = 0
    line = 0
    while line < n_line:
        n_atom = int(xyz[line])
        meta = xyz[line + 1]
        symb = [" "] * n_atom
        x = np.zeros((n_atom, 3))
        l2 = line + 2
        for l in range(0, n_atom):
            lst = (" ".join(xyz[l + l2].split())).split(" ")
            symb[l] = lst[0]
            x[l] = np.array(lst[1:4])
        if blk == 0:
            xyz_db = [[symb, x]]
            meta_db = [[meta]]
        else:
            xyz_db.append([symb, x])
            meta_db.append([meta])


        line += n_atom + 2
        blk += 1

    return xyz_db, blk, meta_db


def sqldb_parse_xyz(name, fid=None, nameset=None, xid=None, ethsd=None, temp=None, posT=True, excludexid=False, sort_ids=False):
    """ Reads SQLite xyz database file and returns a list of lists, each of the latter containing two items:
		1) a list of strings, being chemical symbols of each atom in the configuration
		   e.g. ["H","H","O"] for a configuration with 2 H and 1 O atoms
		2) a 2d numpy array with n_atom rows and 3 columns, where
		   n_atom is the number of atoms in the configuration, and where
		   each row is the (x,y,z) coordinates of an atom in the configuration
		This is intended to be a means to read a file that's a concatenatenation of many xyz files
		each block can be a different configuration (e.g. different molecule) or the same config but in
		a different geometry
		If a config contains an atom not in the list of types, we stop later
	"""

    if posT:
        if temp != None:
            print('points at {} K will be loaded'.format(temp))
            posT = False
        else:
            temp = 0
            print('all points with positive temperature will be loaded')
    if nameset != None:
        print('Data is only from the db with the names: \n', nameset)
    if xid is not None:
        print('xyz id is preset')
        try:
            if '/' in xid[0]:
                xid = [x.split('/')[-1] for x in xid]
        except:
            pass

    config = name.split('/')[-1].split('.')[0]
    atom = Atoms(config)
    symb = atom.get_chemical_symbols()

    xyz_db = []
    meta_db = []
    idtest = []
    rdb.preamble()
    with rdb.create_connection(name) as conn:
        crsr = conn.cursor()
        if xid == None:
            if nameset == None:
                if ethsd == None:
                    if temp == None:
                        if fid == None:
                            sql_query = f'SELECT xyz.geom, xyz.name, xyz.id FROM xyz;'
                            record = crsr.execute(sql_query)  # execute the filtering
                        else:
                            sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id;'
                            record = crsr.execute(sql_query, str(fid))  # execute the filtering

                    else:
                        if posT:
                            if fid == None:
                                sql_query = f'SELECT xyz.geom, xyz.name, xyz.id FROM xyz WHERE xyz.temp>?;'
                                record = crsr.execute(sql_query, (str(temp),))  # execute the filtering
                            else:
                                sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE energy.fidelity=? AND xyz.temp>? AND xyz.id=energy.xyz_id;'
                                record = crsr.execute(sql_query, (str(fid), temp))  # execute the filtering
                        else:
                            if fid == None:
                                sql_query = f'SELECT xyz.geom, xyz.name, xyz.id FROM xyz WHERE xyz.temp=?;'
                                record = crsr.execute(sql_query, (str(temp),))  # execute the filtering
                            else:
                                sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE energy.fidelity=? AND xyz.temp=? AND xyz.id=energy.xyz_id;'
                                record = crsr.execute(sql_query, (str(fid), temp))  # execute the filtering
                else:
                    if temp == None:
                        sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id AND energy.E<?;'
                        record = crsr.execute(sql_query, (str(fid), ethsd))  # execute the filtering
                    else:
                        if posT:
                            sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id AND energy.E<? AND xyz.temp>?;'
                            record = crsr.execute(sql_query, (str(fid), ethsd, temp))  # execute the filtering
                        else:
                            sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id AND energy.E<? AND xyz.temp=?;'
                            record = crsr.execute(sql_query, (str(fid), ethsd, temp))  # execute the filtering
                for r in record:
                    xyz_db.append([symb, np.array(r['geom'])])
                    if fid == None:
                        meta_db.append([r['id'], r['name'], 0, np.array([['label', '{}/{}'.format(config, r['id'])]], dtype='<U12'), 0, 0])
                    else:
                        meta_db.append([r['id'], r['name'], r['calc'], r['calc_params'], r['Force'], r['E']])


            else:
                if ethsd == None:
                    if temp == None:
                        sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE xyz.name LIKE ? AND energy.fidelity=? AND xyz.id=energy.xyz_id;'
                        for nm in nameset:
                            record = crsr.execute(sql_query, (nm, str(fid)))  # execute the filtering
                            for r in record:
                                xyz_db.append([symb, np.array(r['geom'])])
                                meta_db.append([r['id'], r['name'], r['calc'], r['calc_params'], r['Force'], r['E']])
                                idtest.append(r['id'])
                    else:
                        if posT:
                            sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE xyz.name LIKE ? AND energy.fidelity=? AND xyz.temp>? AND xyz.id=energy.xyz_id;'
                        else:
                            sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE xyz.name LIKE ? AND energy.fidelity=? AND xyz.temp=? AND xyz.id=energy.xyz_id;'
                        for nm in nameset:
                            record = crsr.execute(sql_query, (nm, str(fid), temp))  # execute the filtering
                            for r in record:
                                xyz_db.append([symb, np.array(r['geom'])])
                                meta_db.append([r['id'], r['name'], r['calc'], r['calc_params'], r['Force'], r['E']])
                                idtest.append(r['id'])
                else:
                    if temp == None:
                        sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE xyz.name LIKE ? AND energy.fidelity=? AND energy.E<? AND xyz.id=energy.xyz_id;'
                        for nm in nameset:
                            record = crsr.execute(sql_query, (nm, str(fid), ethsd))  # execute the filtering
                            for r in record:
                                xyz_db.append([symb, np.array(r['geom'])])
                                meta_db.append([r['id'], r['name'], r['calc'], r['calc_params'], r['Force'], r['E']])
                                idtest.append(r['id'])
                    else:
                        if posT:
                            sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE xyz.name LIKE ? AND energy.fidelity=? AND energy.E<? AND xyz.temp>? AND xyz.id=energy.xyz_id;'
                        else:
                            sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE xyz.name LIKE ? AND energy.fidelity=? AND energy.E<? AND xyz.temp=? AND xyz.id=energy.xyz_id;'
                        for nm in nameset:
                            record = crsr.execute(sql_query, (nm, str(fid), ethsd, temp))  # execute the filtering
                            for r in record:
                                xyz_db.append([symb, np.array(r['geom'])])
                                meta_db.append([r['id'], r['name'], r['calc'], r['calc_params'], r['Force'], r['E']])
                                idtest.append(r['id'])

        else:
            if excludexid:
                if nameset == None:
                    if temp is None:
                        sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.id NOT IN (' + ','.join(
                            map(str, xid)) + ')'
                        record = crsr.execute(sql_query, str(fid))  # execute the filtering
                    else:
                        if posT:
                            sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.temp>? AND xyz.id NOT IN (' + ','.join(
                                map(str, xid)) + ')'
                        else:
                            sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.temp=? AND xyz.id NOT IN (' + ','.join(
                                map(str, xid)) + ')'
                        record = crsr.execute(sql_query, (str(fid), temp))  # execute the filtering
                    for r in record:
                        xyz_db.append([symb, np.array(r['geom'])])
                        meta_db.append([r['id'], r['name'], r['calc'], r['calc_params'], r['Force'], r['E']])
                        idtest.append(r['id'])
                else:
                    if temp == None:
                        sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE xyz.name LIKE ? AND energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.id NOT IN (' + ','.join(
                                    map(str, xid)) + ')'
                        for nm in nameset:
                            record = crsr.execute(sql_query, (nm, str(fid)))  # execute the filtering
                            for r in record:
                                xyz_db.append([symb, np.array(r['geom'])])
                                meta_db.append([r['id'], r['name'], r['calc'], r['calc_params'], r['Force'], r['E']])
                                idtest.append(r['id'])
                    else:
                        if posT:
                            sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE xyz.name LIKE ? AND energy.fidelity=? AND xyz.temp>? AND xyz.id=energy.xyz_id AND xyz.id NOT IN (' + ','.join(
                                    map(str, xid)) + ')'
                        else:
                            sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE xyz.name LIKE ? AND energy.fidelity=? AND xyz.temp=? AND xyz.id=energy.xyz_id AND xyz.id NOT IN (' + ','.join(
                                    map(str, xid)) + ')'
                        for nm in nameset:
                            record = crsr.execute(sql_query, (nm, str(fid), temp))  # execute the filtering
                            for r in record:
                                xyz_db.append([symb, np.array(r['geom'])])
                                meta_db.append([r['id'], r['name'], r['calc'], r['calc_params'], r['Force'], r['E']])
                                idtest.append(r['id'])

            else:
                if fid is not None:
                    if temp is None:
                        sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.id IN (' + ','.join(
                            map(str, xid)) + ')'
                        record = crsr.execute(sql_query, str(fid))  # execute the filtering
                    else:
                        if posT:
                            sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.temp>? AND xyz.id IN (' + ','.join(
                                map(str, xid)) + ')'
                        else:
                            sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.temp=? AND xyz.id IN (' + ','.join(
                                map(str, xid)) + ')'
                        record = crsr.execute(sql_query, (str(fid), temp))  # execute the filtering
                    for r in record:
                        xyz_db.append([symb, np.array(r['geom'])])
                        meta_db.append([r['id'], r['name'], r['calc'], r['calc_params'], r['Force'], r['E']])
                else:
                    if temp is None:
                        sql_query = f'SELECT xyz.geom, xyz.name, xyz.id FROM xyz WHERE xyz.id IN (' + ','.join(
                            map(str, xid)) + ')'
                        record = crsr.execute(sql_query)  # execute the filtering
                    else:
                        if posT:
                            sql_query = f'SELECT xyz.geom, xyz.name, xyz.id FROM xyz WHERE xyz.temp>? AND xyz.id IN (' + ','.join(
                                map(str, xid)) + ')'
                        else:
                            sql_query = f'SELECT xyz.geom, xyz.name, xyz.id FROM xyz WHERE xyz.temp=? AND xyz.id IN (' + ','.join(
                                map(str, xid)) + ')'
                        record = crsr.execute(sql_query, (temp,))  # execute the filtering
                    for r in record:
                        xyz_db.append([symb, np.array(r['geom'])])
                        meta_db.append(
                            [r['id'], r['name'], 0, np.array([['label', '{}/{}'.format(config, r['id'])]], dtype='<U12'), 0,
                             0])

    blk = len(xyz_db)
    if sort_ids:
        print('sorting by ids')
        ids = np.array([i[0] for i in meta_db])
        indx_sort = sorted(range(len(ids)), key=ids.__getitem__)
        xyz_db = [xyz_db[i] for i in indx_sort]
        meta_db = [meta_db[i] for i in indx_sort]#meta_db[indx_sort]
    

    return xyz_db, blk, meta_db


def sqldb_parse_xyz_all(database, fid=None, names=None, returnE=False, temp=None):
    """ Reads SQLite xyz database file and returns a list of lists. """

    t0 = timeit.default_timer()
    atom = Atoms(database.split('/')[-1].split('.')[0])
    symb = atom.get_chemical_symbols()
    xyz_db = []
    meta_db = []
    rdb.preamble()
    with rdb.create_connection(database) as conn:
        crsr = conn.cursor()
        if names != None:
            if not returnE:
                if temp != None:
                    sql_query = f'SELECT xyz.geom, xyz.name, xyz.id FROM xyz WHERE xyz.name LIKE ? AND xyz.temp=?;'
                else:
                    sql_query = f'SELECT xyz.geom, xyz.name, xyz.id FROM xyz WHERE xyz.name LIKE ?;'
            else:
                if temp != None:
                    sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E FROM xyz, energy WHERE xyz.name LIKE ? AND energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.temp=?;'
                else:
                    sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E FROM xyz, energy WHERE xyz.name LIKE ? AND energy.fidelity=? AND xyz.id=energy.xyz_id;'

            for name in names:
                if returnE:
                    if temp != None:
                        record = crsr.execute(sql_query, (name, str(fid), temp))  # execute the filtering
                    else:
                        record = crsr.execute(sql_query, (name, str(fid)))  # execute the filtering
                else:
                    if temp != None:
                        record = crsr.execute(sql_query, (name, temp))  # execute the filtering
                    else:
                        record = crsr.execute(sql_query, [name])  # execute the filtering
                for training in record:
                    xyz_db.append([symb, np.array(training['geom'])])
                    if returnE:
                        meta_db.append([training['id'], training['name'], 0, 0, training['E']])
                    else:
                        meta_db.append([training['id'], training['name'], 0, 0, 0])
        else:
            if not returnE:
                if temp != None:
                    sql_query = f'SELECT xyz.geom, xyz.id FROM xyz WHERE xyz.temp=?;'
                else:
                    sql_query = f'SELECT xyz.geom, xyz.id FROM xyz;'
            else:
                if temp != None:
                    sql_query = f'SELECT xyz.geom, xyz.id, energy.E FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.temp=?;'
                else:
                    sql_query = f'SELECT xyz.geom, xyz.id, energy.E FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id;'

            if returnE:
                if temp != None:
                    record = crsr.execute(sql_query, (str(fid), temp))  # execute the filtering
                else:
                    record = crsr.execute(sql_query, (str(fid)))  # execute the filtering
            else:
                if temp != None:
                    record = crsr.execute(sql_query, (temp))  # execute the filtering
                else:
                    record = crsr.execute(sql_query)  # execute the filtering
            for training in record:
                xyz_db.append([symb, np.array(training['geom'])])
                if returnE:
                    meta_db.append([training['id'], training['name'], 0, 0, training['E']])
                else:
                    meta_db.append([training['id'], training['name'], 0, 0, 0])
    blk = len(xyz_db)
    print('nblk:', blk)
    t1 = timeit.default_timer()
    print('Time to read sqldb: ', t1 - t0)

    return xyz_db, blk, meta_db

def write_tvtmsk_xyzid(dpes, dbname, sname=None, fidlevel=0, trxid=None, testxid=None, temp=0, nameset=None, uniformsamp=False):
    print('In write_tvtmsk_xyzid')
    if os.path.exists(sname):
        print('!!! Warning !!! \nyou already have the file: {} \nnew xyzid list will be appended to the existing file.'.format(sname))

    if fidlevel == None:
        print('fidelity level is not specified.')
        sys.exit()
    else:
        fidlevel = str(fidlevel)
    if isinstance(dbname, list):
        dbnames = dbname
    else:
        dbnames = [dbname]

    if testxid is None:
        ids_all = [dpes.meta[i][-2] for i in range(dpes.ndat)]
        for dbnm in dbnames:
            config = dbnm.split('/')[-1].split('.')[0]
            if not '/' in ids_all[0] and len(dbnames) > 1:
                print('molecule is not specified for a given xyzid list. please regenerate the hdf5 file using the updated script.')
                sys.exit()
            else:
                if '/' not in ids_all[0]:
                    ids = ids_all
                else:
                    ids = [molid.split('/')[1] for molid in ids_all if molid.split('/')[0] == config]
                idE_xyz = []
                rdb.preamble()
                with rdb.create_connection(dbnm) as conn:
                    crsr = conn.cursor()
                    if temp == 0:
                        sql_query = f'SELECT xyz.id, energy.E FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.id IN (' + ','.join(
                            map(str, ids)) + ')'
                        record = crsr.execute(sql_query, (fidlevel,))  # execute the filtering
                    else:
                        sql_query = f'SELECT xyz.id, energy.E FROM xyz, energy WHERE xyz.temp=? AND energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.id IN (' + ','.join(
                            map(str, ids)) + ')'
                        record = crsr.execute(sql_query, (temp, fidlevel))  # execute the filtering

                    for r in record:
                        idE_xyz.append(['{}/{}'.format(config, r['id']), r['E']])

                ids_sql = [id[0].split('/')[-1] for id in idE_xyz]
                if ids_sql != ids:
                    idE_xyz2 = []
                    if '/' in idE_xyz[0][0]:
                        for id in ids:
                            idE_xyz2.extend([idE for idE in idE_xyz if idE[0].split('/')[-1] == id])
                    else:
                        for id in ids:
                            idE_xyz2.extend([idE for idE in idE_xyz if idE[0] == id])
                    idE_xyz = idE_xyz2.copy()

                if '{}' in sname:
                    sname = sname.format(len(idE_xyz)) 
                f = open(sname, "a")
                write_2s=True
                with f:
                    print("write_tvt_mask: writing file: ... ", sname, end='')
                    if write_2s:
                        for i in range(0, len(idE_xyz)):
                            if dpes.tvtmsk[i] == -1:
                                print('{} {:3d} {:022.14e}'.format(idE_xyz[i][0], dpes.tvtmsk[i], idE_xyz[i][1]), file=f)
                            else:
                                print('{} {:3d} {:022.14e}'.format(idE_xyz[i][0], 2, idE_xyz[i][1]), file=f)
                    else:
                        for i in range(0, dpes.ndat):
                            print('{} {:3d} {:022.14e}'.format(idE_xyz[i][0], dpes.tvtmsk[i], idE_xyz[i][1]), file=f)
                    print(" done")

    else:
        print("write_tvt_mask: reading from sqlite db:",dbnames)
        for dbnm in dbnames:
            config = dbnm.split('/')[-1].split('.')[0]
            if not '/' in testxid[0] and len(dbnames) > 1:
                print('molecule is not specified for a given test xyzid list. please check the xyzid list file.')
                sys.exit()
            else:
                if '/' not in testxid[0]:
                    ids = testxid
                else:
                    ids = [molid.split('/')[1] for molid in testxid if molid.split('/')[0] == config]

            idE_tst = []
            rdb.preamble()
            with rdb.create_connection(dbnm) as conn:
                crsr = conn.cursor()
                if temp == 0:
                    sql_query = f'SELECT xyz.id, energy.E FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.id IN (' + ','.join(
                        map(str, ids)) + ')'
                    record = crsr.execute(sql_query, (fidlevel,))  # execute the filtering
                else:
                    sql_query = f'SELECT xyz.id, energy.E FROM xyz, energy WHERE xyz.temp=? AND energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.id IN (' + ','.join(
                        map(str, ids)) + ')'
                    record = crsr.execute(sql_query, (temp, fidlevel))  # execute the filtering

                for r in record:
                    idE_tst.append(['{}/{}'.format(config, r['id']), r['E']])

            idE_tr = []
            if trxid is None:
                # get unique name for entire dataset
                tmp = []
                if nameset is None:
                    rdb.preamble()
                    with rdb.create_connection(dbnm) as conn:
                        crsr = conn.cursor()
                        sql_query = f'SELECT xyz.name FROM xyz WHERE xyz.dist=0'
                        record = crsr.execute(sql_query)  # execute the filtering

                        for r in record:
                            tmp.append([r['name']])
                else:
                    for nm in nameset:
                        with rdb.create_connection(dbnm) as conn:
                            crsr = conn.cursor()
                            sql_query = f'SELECT xyz.name FROM xyz WHERE xyz.dist=0 AND xyz.name LIKE ?'
                            record = crsr.execute(sql_query, (nm,))  # execute the filtering
                            for r in record:
                                tmp.append([r['name']])

                uniqnms = np.unique(tmp)

                if uniformsamp:
                    nsamp = 100
                    npath = 20
                    print('{}: training points are selected randomly with nsamp={} and npath={}'.format(config, nsamp,
                                                                                                        npath))
                for nm in uniqnms:
                    # print('temp: {}, fidlevel: {}, nm: {}'.format(temp, fidlevel, nm))
                    if uniformsamp:
                        idE_tr_nm = []
                        rdb.preamble()
                        with rdb.create_connection(dbnm) as conn:
                            crsr = conn.cursor()
                            if temp == 0:
                                sql_query = f'SELECT xyz.name, xyz.id, energy.E FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.name=? AND xyz.id NOT IN (' + ','.join(
                                    map(str, ids)) + ') LIMIT ?'
                                if 'irc' in nm:
                                    record = crsr.execute(sql_query, (fidlevel, nm, npath))  # execute the filtering
                                else:
                                    record = crsr.execute(sql_query, (fidlevel, nm, nsamp))  # execute the filtering
                            else:
                                sql_query = f'SELECT xyz.name, xyz.id, energy.E FROM xyz, energy WHERE xyz.temp=? AND energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.name=? AND xyz.id NOT IN (' + ','.join(
                                    map(str, ids)) + ') LIMIT ?'
                                if 'irc' in nm:
                                    record = crsr.execute(sql_query, (temp, fidlevel, nm, npath))  # execute the filtering
                                else:
                                    record = crsr.execute(sql_query, (temp, fidlevel, nm, nsamp))  # execute the filtering
                            for r in record:
                                idE_tr_nm.append(['{}/{}'.format(config, r['id']), r['E']])
                            if 'irc' in nm:
                                if len(idE_tr_nm) < npath:
                                    print('!!! Warning !!! \nNot enough data points for {} in {}\nrequired points: {}\navailable points: {}'.format(nm, config, npath, len(idE_tr_nm)))
                                    sys.exit()
                                else:
                                    idE_tr.extend(idE_tr_nm)
                            else:
                                if len(idE_tr_nm) < nsamp:
                                    print('!!! Warning !!! \nNot enough data points for {} in {}\nrequired points: {}\navailable points: {}'.format(nm, config, nsamp, len(idE_tr_nm)))
                                    sys.exit()
                                else:
                                    idE_tr.extend(idE_tr_nm)
                    else:
                        rdb.preamble()
                        with rdb.create_connection(dbnm) as conn:
                            crsr = conn.cursor()
                            if temp == 0:
                                sql_query = f'SELECT xyz.name, xyz.id, energy.E FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.name=? AND xyz.id NOT IN (' + ','.join(
                                    map(str, ids)) + ')'
                                record = crsr.execute(sql_query, (fidlevel, nm))  # execute the filtering
                            else:
                                sql_query = f'SELECT xyz.name, xyz.id, energy.E FROM xyz, energy WHERE xyz.temp=? AND energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.name=? AND xyz.id NOT IN (' + ','.join(
                                    map(str, ids)) + ')'
                                record = crsr.execute(sql_query, (temp, fidlevel, nm))  # execute the filtering

                            for r in record:
                                idE_tr.append(['{}/{}'.format(config, r['id']), r['E']])


            else:
                if not '/' in trxid[0] and len(dbnames) > 1:
                    print(
                        'molecule is not specified for a given training xyzid list. please check the xyzid list file.')
                    sys.exit()
                else:
                    if '/' not in trxid[0]:
                        ids = trxid
                    else:
                        ids = [molid.split('/')[1] for molid in trxid if molid.split('/')[0] == config]

                rdb.preamble()
                with rdb.create_connection(dbnm) as conn:
                    crsr = conn.cursor()
                    if temp == 0:
                        sql_query = f'SELECT xyz.id, energy.E FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.id IN (' + ','.join(
                            map(str, ids)) + ')'
                        record = crsr.execute(sql_query, fidlevel)  # execute the filtering
                    else:
                        sql_query = f'SELECT xyz.id, energy.E FROM xyz, energy WHERE xyz.temp=? AND energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.id IN (' + ','.join(
                            map(str, ids)) + ')'
                        record = crsr.execute(sql_query, (temp, fidlevel))  # execute the filtering
                    for r in record:
                        idE_tr.append(['{}/{}'.format(config, r['id']), r['E']])

            # print(len(idE_tr))
            if '{}' in sname:
                sname = sname.format(len(idE_tr))
            # print(sname)

            f = open(sname, "a")
            with f:
                print("write_tvt_mask: writing file: ... ", sname, end='')
                for i in range(0, idE_tr.__len__()):
                    print('{} {:3d} {:022.14e}'.format(idE_tr[i][0], 2, idE_tr[i][1]), file=f)
                for i in range(0, idE_tst.__len__()):
                    print('{} {:3d} {:022.14e}'.format(idE_tst[i][0], -1, idE_tst[i][1]), file=f)
                print(" done")
    return sname





# ====================================================================================================

# ====================================================================================================
def parse_xyz(name):
    """ Reads xyz file and returns a list containing two items:
		1) a list of strings, being chemical symbols of each atom in the system
		2) a 2d numpy array with n_atom rows and 3 columns where
		   each row is the (x,y,z) coordinates of an atom in the system
	"""
    try:
        f = open(name, "r")
    except IOError:
        print("Could not open file:" + name)
    # sys.exit()
    with f:
        xyz = f.readlines()

    n_line = len(xyz)
    n_atom = int(xyz[0])
    assert (n_line == n_atom + 2)

    symb = [" "] * n_atom
    x = np.zeros((n_atom, 3))

    for line in range(2, n_line):
        lst = (" ".join(xyz[line].split())).split(" ")
        symb[line - 2] = lst[0]
        x[line - 2] = np.array(lst[1:4])

    return symb, x


# ====================================================================================================
# ====================================================================================================
def read_xyz(name):
    """
	 Reads an xyz file using parse_xyz() and repackages the 2d array of
	 atom positions returned from it, converting it from a n_atom x 3 2d array
	 to a 1 x 3n_atom 2d array
	"""
    symb, x = parse_xyz(name)
    x = np.array([x.flatten()])

    return symb, x


# ====================================================================================================
def read_and_append_xyz(name, symb, x):
    symb_new, x_new = read_xyz(name)
    assert (checkEqual(symb, symb_new))
    x = np.concatenate((x, x_new))

    return symb, x


def read_and_append_xyz_torch(name, symb, x):
    symb_new, x_new = read_xyz(name)
    assert (checkEqual(symb, symb_new))
    x = np.concatenate((x, x_new))

    return symb, torch.tensor(x)


def append_xyz_torch(conf, symb, x):
    symb_new = conf.get_chemical_symbols()
    x_new = torch.tensor([conf.get_positions().flatten()])
    assert (checkEqual(symb, symb_new))
    x = np.concatenate((x, x_new))

    return symb, torch.tensor(x)


def append_xyz(conf, symb, x):
    symb_new = conf.get_chemical_symbols()
    x_new = np.array([conf.get_positions().flatten()])
    assert (checkEqual(symb, symb_new))
    x = np.concatenate((x, x_new))

    return symb, x


# ====================================================================================================
# ====================================================================================================
# output AEV db to file
# includes one block for each of the blocks in the db xyz file

def db_out2file(symb, x, y, blk, all_in_one=True):
    if blk == 0:
        tag = 'w'
    else:
        tag = 'a'

    if all_in_one:
        # outputs the AEV of all atoms in each block's config, one row for each atom
        # each column is one element of the AEV
        # output is like this for each block
        # <natm: number_of_atoms_in_configuration> <naev: size_of_AEV>
        # <symbols string in configuration (e.g. HHO)>
        # AEV_0_for_atom_0 AEV_1_for_atom_0 ... AEV_{naev-1}_for_atom_0
        # AEV_0_for_atom_1 AEV_1_for_atom_1 ... AEV_{naev-1}_for_atom_1
        # ...
        # AEV_0_for_atom_{natm-1} AEV_1_for_atom_{natm-1} ... AEV_{naev-1}_for_atom_{natm-1}
        for p in range(0, x.shape[0]):
            with open('aev.' + ''.join(symb) + '.dat', tag) as f:
                print(len(symb), " ", y.shape[2], file=f)
                print(''.join(symb), file=f)
                [np.savetxt(f, [y[p][i]], delimiter=" ", newline='', comments='', footer="\n") for i in
                 range(0, len(symb))]

    else:
        # this option writes one AEV file per atom in the configuration
        # each file name has the atom index in its name
        # each block structure is:
        # <naev: size_of_AEV>
        # <symbols string in configuration (e.g. HHO)>
        # AEV_0_for_this_atom
        # AEV_1_for_this_atom
        # ...
        # AEV_{naev-1}_for_this_atom_{natm-1}
        for p in range(0, x.shape[0]):
            for i in range(0, len(symb)):
                with open('aev.' + ''.join(symb) + '.' + str(i).zfill(len(str(len(symb) - 1))) + '.dat', tag) as f:
                    print(y.shape[2], file=f)
                    print(''.join(symb), file=f)
                    np.savetxt(f, y[p][i])


def multiple_sqldb_parse_xyz(names, fid=None, nameset=None, temp=None, xid=None, excludexid=False, posT=True, sort_ids=False):
    xyz_db = []
    nblk = 0
    meta_db = []
    meta_dist = []
    try:
        if '/' in xid[0]:
            xid_sep = [molid.split('/') for molid in xid]
        else:
            xid_sep = xid
    except:
        pass
    for name in names:
        if xid is not None:
            mol = name.split('/')[-1].split('.')[0]
            xidnew = [id[1] for id in xid_sep if id[0] == mol]
            xyz_db_temp, nblk_temp, meta_db_temp = sqldb_parse_xyz(name, fid=fid, nameset=nameset, temp=temp, xid=xidnew,
                                                                   excludexid=excludexid, posT=posT, sort_ids=sort_ids)
            #xyz_db_temp, nblk_temp, meta_db_temp, dist = sqldb_parse_xyz(name, fid=fid, nameset=nameset, temp=temp, xid=xidnew,
            #                                                       excludexid=excludexid, posT=posT)
        else:
            xyz_db_temp, nblk_temp, meta_db_temp = sqldb_parse_xyz(name, fid=fid, nameset=nameset, temp=temp, posT=posT, sort_ids=sort_ids)
            #xyz_db_temp, nblk_temp, meta_db_temp, dist = sqldb_parse_xyz(name, fid=fid, nameset=nameset, temp=temp, posT=posT)
        xyz_db.extend(xyz_db_temp)
        meta_db.extend(meta_db_temp)
        nblk = nblk + nblk_temp

    return xyz_db, nblk, meta_db



