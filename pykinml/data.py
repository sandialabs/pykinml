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

from pykinml import aev
from pykinml import rdb
try:
    import aevmod
except ModuleNotFoundError:
    pass

from pykinml import hdf5_handler as hd

verbose  = False
vverbose = False

home = Path.home()



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
            atom_types = self.atom_types#['C', 'H']


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
            # print('AEV derivatives were calculated.')
        except:
            pass
            #print('AEV derivatives were not calculated. Please check if aevmod is available.')

        return myaev

    # ==============================================================================================


    def get_data(self, args, xid=None, fid=1, get_aevs=True):

        myaev = None

        if args.input_data_type == 'aev':
            # read AEV data base hdf5 file
            print("Reading aev hdf5 data base (currently implements one input file only)")

            hd.read_aev_db_hdf(self, args.input_data_fname, args.ni, args.nf)
            hd.unpack_data(self, fdata=args.floss)

            print("done reading aev db ...")

        elif args.input_data_type == 'pca':

            # read AEV PCA data base hdf5 file
            print("Reading aev pca hdf5 data base (currently implements one input file only)")

            hd.read_pca_aev_db_hdf(self, args.input_data_fname, args.ni, args.nf)
            hd.unpack_data(self)

            print("done reading aev pca db ...")

        elif args.input_data_type == 'sqlite':
            # define list of atom types in system
            atom_types = args.present_elements

            # set values for radial and angular symmetry function parameters

            nrho_rad = args.aev_params[0]  # number of radial shells in the radial AEV
            nrho_ang = args.aev_params[1]  # number of radial shells in the angular AEV
            nalpha = args.aev_params[2]    # number of angular wedges dividing [0,pi] in the angular AEV
            R_c = args.cuttoff_radius
            # instantiate AEV for given atom types, for given
            try:
                import aevmod
                myaev = aevmod.aev(atom_types, nrho_rad, nrho_ang, nalpha, R_c, beta=args.beta)
            except:
                myaev = aev.Aev(atom_types, nrho_rad, nrho_ang, nalpha, R_c)
                # set the dimension of the data vector being the input for each NN in the system
                self.dimdat = myaev.dout
                print("Constructed aev, output dimensionality is:", myaev.dout)

            # init data class with list of atom types in system
            self.initialize(atom_types)



            # read SQLite data base file 
            names = glob.glob(args.input_data_fname)
            print('names: ', names)
            print('len(names): ', len(names))
            try:
                print('extracting data at temp=', args.temp)
            except:
                args.temp=None
            try:
                print('extracting data for nameset: ', args.nameset)
            except:
                args.nameset=None
            try:
                print('Extracting data for delta learning: ', args.delta)
            except:
                args.delta=False

            if len(names) > 1:
                print("parsing multiple SQLite xyz databases")
                #if args.delta == True:
                xyz_db, nblk, meta_db = multiple_sqldb_parse_xyz(names, fid=fid, nameset=args.nameset,
                                                                 xid=xid, temp=args.temp, sort_ids=args.delta)
                print('SQLite xyz data was extracted, fid: {}'.format(fid))
                if args.delta:
                    xyz_db_lf, nblk_lf, meta_db_lf = multiple_sqldb_parse_xyz(names, fid=args.fidlevel_lf, nameset=args.nameset,
                                                                              xid=xid, temp=args.temp, sort_ids=args.delta)
                    print('SQLite xyz data was extracted, fid: {}'.format(args.fidlevel_lf))
                    for i in range(len(meta_db)):
                        meta_db[i][-1] -= meta_db_lf[i][-1]
                        meta_db[i][-2] -= meta_db_lf[i][-2]

            else:
                #print('args.fidlevel: ', args.fidlevel)
                print("parsing SQLite xyz data base", args.input_data_fname)
                try:
                    if '/' in xid[0]:
                        xid_sep = [molid.split('/')[1] for molid in xid]
                    else:
                        xid_sep = xid
                except:
                    xid_sep = xid
                xyz_db, nblk, meta_db = sqldb_parse_xyz(args.input_data_fname, fid=fid,
                                                        nameset=args.nameset, xid=xid_sep, temp=args.temp, sort_ids=args.delta)
                if args.delta:
                    xyz_db_lf, nblk_lf, meta_db_lf = sqldb_parse_xyz(args.input_data_fname, fid=args.fidlevel_lf,
                                                        nameset=args.nameset, xid=xid_sep, temp=args.temp, sort_ids=args.delta)
                    if args.delta:
                        for i in range(len(meta_db[-1])):
                            meta_db[-1][i] -= meta_db_lf[-1][i]
                            meta_db[-2][i] -= meta_db_lf[-2][i]


            print('in get_data: nblk:', nblk)
            parsed = parse_meta_db(meta_db)
            #print('in get_data: got parsed[0]:',parsed[0])
            tag = None  # tag = 'b3lyp/6-31G'
            con = None  # con = xyz_db[0][0]
            force = args.wrf
            weights = args.wrw


        # build aev database for available xyz database
        print("Building PES AEV data base")
        del meta_db
        if get_aevs:
            self.xyz_to_aev_db(xyz_db, nblk, parsed, myaev, tag, con, force=force, weights=weights, verbose=False)
        else:
            self.get_xdat(xyz_db, nblk,  parsed, force=force)
        if force:
            try:
                print('Building derivative of AEV data base')
                self.xyz_to_daev_db(xyz_db, nblk, myaev)
            except:
                print('AEV derivatives were not calculated. Please check if aevmod is available.')


        print("done...")
        print("ndat:", self.ndat)
        #print("dimdat:", self.dimdat)


        return myaev

    def get_xdat(self, xyz_db, nblk, parsed, force):
        self.ndat=0
        full_energy_data = []
        if force:
            full_force_data = []
        self.full_symb_data = []
        self.pdat = []
        self.meta = []

        for blk in range(0, nblk):
            symb = xyz_db[blk][0]
            energy = parsed[blk][0]
            full_energy_data.append(energy)
            if force:
                full_force_data.append(parsed[blk][-1].flatten())   # HNN 7/31/22: changed from full_force_data.append(parsed[blk][-2].flatten())
            self.full_symb_data.append(symb)
            self.ndat = self.ndat + 1
            #self.pdat.append(np.reshape(x, (-1, 3)))
            self.meta.append(parsed[blk])

        self.xdat = np.empty([self.ndat, self.num_nn + 1], dtype=object)
        if force:
            self.fdat = np.empty([self.ndat, 1], dtype='float64').tolist()

        for i in range(0, self.ndat):
            xdatnn = [[]] * self.num_nn
            for j, s in enumerate(self.full_symb_data[i]):
                k = self.atom_types.index(s)
                xdatnn[k] = xdatnn[k] + []

            for k in range(0, self.num_nn):
                self.xdat[i][k] = xdatnn[k]

            #print('xyz_to_aev_db: i:',i,', num:',self.num_nn,',E:',full_energy_data[i])
            self.xdat[i][self.num_nn] = [float(full_energy_data[i])]
            if force:
                self.fdat[i] = [full_force_data[i]]
                #print('xyz_to_aev_db: Got force data')
        #self.tvtmsk = np.ones([self.ndat], dtype=int)


        return


    # ==========================================================================================

    
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
        #del self.meta
        aevmodule = myaev.__class__.__module__
        #print('aev module:', aevmodule)

        if aevmodule != 'aevmod':
            print('No aevmod')
            print('aevmod module requires for Jacobian calculation')
            sys.exit()

        # J_C = []
        # J_H = []
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
        print(d0)
        for i in range(len(self.fdat)):
            d1.append(len(self.fdat[i]))
            for j in range(len(self.fdat[i])):
                d2.append(len(self.fdat[i][j]))
        self.fd2 = d2
        self.padded_fdat = np.zeros((d0, max(d1), max(d2)))
        for i in range(len(self.fdat)):
            for j in range(len(self.fdat[i])):
                self.padded_fdat[i][j][:len(self.fdat[i][j])] = self.fdat[i][j]
        del self.dxdat
        #=====================================================
        return

    def xyz_to_aev_db(self, xyz_db, nblk, parsed, myaev, target_theory=None, target_symb=None, force=True, weights=None, verbose=False):
        """
		Method to build the AEV data base for the Data_pes object
		"""
        aevmodule = myaev.__class__.__module__

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

            energy = parsed[blk][0]   # HNN 7/31/22: changed from: parsed[blk][-1]

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
                    if vverbose:
                        with np.printoptions(precision=4, suppress=True):
                            print("Configuration:", symb)
                            print("x:", [xp.tolist() for xp in np.reshape(x, (-1, 3))])

                conf.set_index_sets(*myaev.bld_index_sets(symb))
                y = myaev.eval(symb, *conf.get_index_sets(), x)[0]  # evaluate AEV

            full_aev_data.append(y)
            #print('full_aev_data: ', full_aev_data)
            full_energy_data.append(energy)
            if force:
                full_force_data.append(parsed[blk][-1].flatten())   # HNN 7/31/22: changed from full_force_data.append(parsed[blk][-2].flatten()) 
            self.full_symb_data.append(symb)
            self.ndat = self.ndat + 1
            self.pdat.append(np.reshape(x, (-1, 3)))
            self.meta.append(parsed[blk])


        self.dimdat = full_aev_data[0].shape[-1]


        # ==========================================================================================
        # prep data for training and testing

        # for i in range(0,ndat):
        #	print (i)
        #	print(full_aev_data[i],full_energy_data[i])

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
                # nuniq = list(set(names))
                # dicnm = dict(zip(nuniq, list(range(1, len(nuniq) + 1))))
                # zid = [dicnm[v] for v in names]
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
        #self.tvtmsk = np.ones([self.ndat], dtype=int)
        #self.ntrdat = self.ndat
        #self.nvldat = 0
        #self.ntsdat = 0


        return



    def write_tvt_mask(self, fname, tvtmsk):
        try:
            f = open(fname, "w")
        except IOError:
            print("write_tvt_mask: could not open file:", fname)
        # sys.exit()
        with f:
            print("write_tvt_mask: writing file: ... ", fname, end='')
            for i, msk in enumerate(tvtmsk):
                print('{:9d} {:3d}'.format(i, msk), file=f)
            print(" done")

    def prep_data(self, device='cpu'):
        """
		Method to prepare data by packaging it appropriately for NN batch computations
		"""
        self.irun = [i for i in range(0, self.ndat)]
        self.nat = [[self.full_symb_data[self.irun[i]].count(self.atom_types[t]) for i in range(len(self.irun))] for t in
                      range(self.num_nn)]
        self.nat_maxs = [max(i) for i in self.nat]
        
        self.aevs = [[[] for j in range(self.num_nn)] for i in range(len(self.irun))]
        for i in range(len(self.irun)):
            for j in range(self.num_nn):
                self.aevs[i][j] = torch.tensor(self.xdat[self.irun[i]][j])

        #print(self.dxdat)
        #try:
        self.daevs = [[[] for j in range(self.num_nn)] for i in range(len(self.irun))]
        for i in range(len(self.irun)):
            for j in range(self.num_nn):
                self.daevs[i][j] = torch.tensor([self.dxdat[self.irun[i]][j][k] for k in range(self.nat_maxs[j])])
        #except:
        #    pass
        if device != 'cpu':
            for i in range(len(self.irun)):
                for j in range(self.num_nn):
                    self.aevs[i][j] = self.aevs[i][j].to(device)
                    self.daevs[i][j] = self.daevs[i][j].to(device)

        return 0



    def prep_training_data(self, train_ind, bpath=None, with_aev_data=True):
        """
                Method to prepare training data by packaging it appropriately for NN batch computations
                """

        itr = [i for i in range(len(self.md)) if self.md[i] in train_ind]

        print('self.num_nn: ', self.num_nn)
        print('len(self.full_symb_data): ', len(self.full_symb_data))
        print('self.atom_types: ', self.atom_types)
        print('max(itr): ', max(itr))
        self.nattr = [[self.full_symb_data[itr[i]].count(self.atom_types[t]) for i in range(len(itr))] for t in
                      range(self.num_nn)]

        nattr_maxs = [max(i) for i in self.nattr]
        print('nattr_maxs: ', nattr_maxs)





        self.train_aevs = [[[] for j in range(self.num_nn)] for i in range(len(itr))]
        self.train_engs = [[] for i in range(len(itr))]


        for i in range(len(itr)):
            if with_aev_data:
                for j in range(self.num_nn):
                    self.train_aevs[i][j] = torch.tensor(self.xdat[itr[i]][j])
            self.train_engs[i] = torch.tensor(self.xdat[itr[i]][-1])
        torch.save(self.train_engs, bpath+'train_engs')
        if with_aev_data:
            torch.save(self.train_aevs, bpath+'train_aevs')
            torch.save(self.dimdat, bpath+'aev_length')

        try:
            self.train_daevs = [[[] for j in range(self.num_nn)] for i in range(len(itr))]
            self.train_forces = [[] for i in range(len(itr))]
            self.train_fdims = [[] for i in range(len(itr))]
            for i in range(len(itr)):
                if with_aev_data:
                    for j in range(self.num_nn):
                        self.train_daevs[i][j] = torch.tensor([self.padded_dxdat[itr[i]][j][k] for k in range(nattr_maxs[j])])
                self.train_forces[i] = torch.tensor([self.padded_fdat[itr[i]][0]])
                self.train_fdims[i] = len(self.fdat[itr[i]][0])
            if with_aev_data:
                torch.save(self.train_daevs, bpath+'train_daevs')
                del self.train_daevs
            torch.save(self.train_forces, bpath+'train_forces')
            torch.save(self.train_fdims, bpath+'train_fdims')
            #del self.train_daevs
            del self.train_forces
            del self.train_fdims
        except:
            pass

        return 0



    def prep_validation_data(self, bpath = None):
        """
		Method to prepare validation data by packaging it appropriately for NN batch computations
                Currently unsued as training/validation sets are processed together and split later
		"""
        print('data.py: prep_validation_data')

        ivl = [i for i in range(0, self.ndat) if self.tvtmsk[i] == 0]


        self.natvl = [[self.full_symb_data[ivl[i]].count(self.atom_types[t]) for i in range(self.nvldat)] for t in
                      range(self.num_nn)]
        
        natvl_maxs = [max(i) for i in self.natvl]


        self.valid_aevs = [[[] for j in range(self.num_nn)] for i in range(len(ivl))]
        self.valid_engs = [[] for i in range(len(ivl))]

        for i in range(len(ivl)):
            for j in range(self.num_nn):
                self.valid_aevs[i][j] = torch.tensor(self.xdat[ivl[i]][j])
            self.valid_engs[i] = torch.tensor(self.xdat[ivl[i]][-1])
        torch.save(self.valid_engs, bpath+'valid_engs')
        torch.save(self.valid_aevs, bpath+'valid_aevs')

        try:
            self.valid_daevs = [[[] for j in range(self.num_nn)] for i in range(len(ivl))]
            self.valid_forces = [[] for i in range(len(ivl))]
            self.valid_fdims = [[] for i in range(len(ivl))]
            for i in range(len(ivl)):
                for j in range(self.num_nn):
                    self.valid_daevs[i][j] = torch.tensor([self.padded_dxdat[ivl[i]][j][k] for k in range(natvl_maxs[j])])
                self.valid_forces[i] = torch.tensor([self.padded_fdat[ivl[i]][0]])
                self.valid_fdims[i] = len(self.fdat[ivl[i]][0])
            torch.save(self.valid_daevs, bpath+'valid_daevs')
            torch.save(self.valid_forces, bpath+'valid_forces')
            torch.save(self.valid_fdims, bpath+'valid_fdims')
        except:
            pass


        return 0

    def prep_testing_data(self, test_ind, bpath=None, with_aev_data=True):
        """
                Method to prepare testing data by packaging it appropriately for NN batch computations
                """


        its = [i for i in range(len(self.md)) if self.md[i] in test_ind]

        self.natts = [[self.full_symb_data[its[i]].count(self.atom_types[t]) for i in range(len(its))] for t in
                      range(self.num_nn)]

        natts_maxs = [max(i) for i in self.natts]


        self.test_aevs = [[[] for j in range(self.num_nn)] for i in range(len(its))]
        self.test_engs = [[] for i in range(len(its))]
        for i in range(len(its)):
            for j in range(self.num_nn):
                if with_aev_data:
                    self.test_aevs[i][j] = torch.tensor(self.xdat[its[i]][j])
            self.test_engs[i] = torch.tensor(self.xdat[its[i]][-1])
        torch.save(self.test_engs, bpath+'test_engs')
        if with_aev_data:
            torch.save(self.test_aevs, bpath+'test_aevs')
        try:
            self.test_daevs = [[[] for j in range(self.num_nn)] for i in range(len(its))]
            self.test_forces = [[] for i in range(len(its))]
            self.test_fdims = [[] for i in range(len(its))]
            for i in range(len(its)):
                if with_aev_data:
                    for j in range(self.num_nn):
                        self.test_daevs[i][j] = torch.tensor([self.padded_dxdat[its[i]][j][k] for k in range(natts_maxs[j])])
                self.test_forces[i] = torch.tensor([self.padded_fdat[its[i]][0]])
                self.test_fdims[i] = len(self.fdat[its[i]][0])
            if with_aev_data:
                torch.save(self.test_daevs, bpath+'test_daevs')
            torch.save(self.test_forces, bpath+'test_forces')
            torch.save(self.test_fdims, bpath+'test_fdims')
        except:
            pass

        return 0




# ====================================================================================================
# check that two lists match exactly
def checkEqual(L1, L2):
    return len(L1) == len(L2) and L1 == L2


# ====================================================================================================


def sqldb_parse_xyz(name, fid=None, nameset=None, xid=None, ethsd=None, temp=None, posT=True, excludexid=False, sort_ids=True):
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
    print('config: ', config)
    atom = Atoms(config)
    symb = atom.get_chemical_symbols()

    # t0 = timeit.default_timer()
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
                            #sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.Force FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.temp=? AND xyz.id IN (' + ','.join(
                            #    map(str, xid)) + ')'
                            sql_query = f'SELECT xyz.geom, xyz.name, xyz.id, energy.E, energy.calc, energy.calc_params, energy.Force FROM xyz, energy WHERE energy.fidelity=? AND xyz.id=energy.xyz_id AND xyz.temp=? AND xyz.id IN (' + ','.join(
                                map(str, xid)) + ')'
                        record = crsr.execute(sql_query, (str(fid), temp))  # execute the filtering
                    for r in record:
                        xyz_db.append([symb, np.array(r['geom'])])
                        #meta_db.append([r['id'], r['name'], r['calc'], r['calc_params'], r['Force'], r['E']])
                        meta_db.append([r['id'], r['name'], 0, np.array([['label', '{}/{}'.format(config, r['id'])]], dtype='<U12'), r['Force'], r['E']])
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
    print('sqldb_parse_xyz: nblk:', blk)

    if sort_ids:
        print('sorting by ids')
        ids = np.array([i[0] for i in meta_db])
        indx_sort = sorted(range(len(ids)), key=ids.__getitem__)
        xyz_db = [xyz_db[i] for i in indx_sort]
        meta_db = [meta_db[i] for i in indx_sort]#meta_db[indx_sort]
    

    return xyz_db, blk, meta_db



# ====================================================================================================


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


# ====================================================================================================

