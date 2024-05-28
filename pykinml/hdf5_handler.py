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

import sys
import os
import timeit

import numpy as np
import torch
import h5py






newhdfg  = True

def pack_data(dpes):
    """
    Method to pack data for array-style storage as needed
    Generates the following numpy arrays
    xit 2d array, ndat rows, num_nn colums, int array
    xft 2d array, ndat rows, num_nn colums, int array
    xip 1d array, ndat elements, int array
    xfp 1d array, ndat elements, int array
    xp  2d array, * rows, 3 columns, float array
    xx  2d array, * rows, laev columns, float
    yy  2d array, ndat rows, 1 column, float
    """

    # nat[i][t] : num of atoms of type t in config i
    nat = np.array([[dpes.full_symb_data[i].count(dpes.atom_types[t]) for t in range(dpes.num_nn)] for i in range(dpes.ndat)])

    # nac[i] : num of atoms of all types in config i
    nac = np.array([np.sum(nat[i]) for i in range(dpes.ndat)])  # atoms in config i

    # total number of atoms of all types in all configurations in the ndat data points
    natot = np.sum(nac)

    # xxi[i] : num of atoms of all types in configs 0, 1, ..., i-1
    xxi = np.array([nat[:i].sum() for i in range(dpes.ndat)])

    # xit[i][t] : num of atoms of all types in configs 0, 1, ... i-1  plus  num of atoms of types 0, 1, ..., t-1 in config i
    # xft[i][t] : num of atoms of all types in configs 0, 1, ... i-1  plus  num of atoms of types 0, 1, ..., t   in config i
    dpes.xit = np.array([[xxi[i] + np.sum(nat[i][:t]) for t in range(dpes.num_nn)] for i in range(dpes.ndat)])
    dpes.xft = np.array([[xxi[i] + np.sum(nat[i][:t + 1]) for t in range(dpes.num_nn)] for i in range(dpes.ndat)])


    ldat = len(dpes.xdat[0][0][0])
    print("pack_data: ldat:", ldat)

    dpes.xx = np.empty([natot, ldat])
    dpes.dxx = np.empty([natot, ldat]).tolist()
    dpes.yy = np.empty([dpes.ndat])
    dpes.ww = np.empty([dpes.ndat])
    dpes.ff = np.empty([dpes.ndat]).tolist()
    tmp_print = True
    for i in range(dpes.ndat):
        for t in range(dpes.num_nn):
            dpes.xx[dpes.xit[i][t]: dpes.xft[i][t]] = np.array(dpes.xdat[i][t])
            try:
                tmp = np.array(dpes.dxdat[i][t])
                dpes.dxx[dpes.xit[i][t]: dpes.xft[i][t]] = tmp.reshape(tmp.shape[0], tmp.shape[1]*tmp.shape[2])
                if tmp_print:
                    print('pack AEV derivative data')
                    tmp_print = False
            except:
                if tmp_print:
                    print('no AEV derivative data was packed')
                    tmp_print = False
        dpes.yy[i:i + 1] = np.array(dpes.xdat[i][dpes.num_nn])
        try:
            dpes.ww[i:i + 1] = dpes.w[i]
        except:
            pass
        try:
            dpes.ff[i] = dpes.fdat[i]
        except:
            pass
    dpes.yy = dpes.yy.reshape((dpes.ndat, 1))
    dpes.ww = dpes.ww.reshape((dpes.ndat, 1))
    try:
        dpes.ff = np.array(dpes.ff, dtype=object)
        print('pack force data')
    except:
        print('no force data was packed')


    # xip[i] : num of atoms of all types in configs 0, 1, ..., i-1
    # xfp[i] : num of atoms of all types in configs 0, 1, ..., i
    dpes.xip = [np.sum(nac[:i]) for i in range(dpes.ndat)]
    dpes.xfp = [dpes.xip[i] + nac[i] for i in range(dpes.ndat)]
    dpes.xp = np.empty([natot, 3])
    for i in range(dpes.ndat):
        dpes.xp[dpes.xip[i]: dpes.xfp[i]] = dpes.pdat[i]


def unpack_data(dpes, fdata=True):
    """
    Unpack data, to build xdat and pdat lists
    """

    fd = [[] for i in range(dpes.ndat)]
    wd = [[] for i in range(dpes.ndat)]
    dxxd = [[[] for t in range(dpes.num_nn)] for i in range(dpes.ndat)]
    xxd = [[[] for t in range(dpes.num_nn + 1)] for i in range(dpes.ndat)]
    tmp_print = True
    for i in range(dpes.ndat):
        for t in range(dpes.num_nn):
            xxd[i][t] = dpes.xx[dpes.xit[i][t]:dpes.xft[i][t]].tolist()
            if fdata:
                try:
                    tmp = np.array(dpes.dxx[dpes.xit[i][t]: dpes.xft[i][t]])
                    dxxdlst = []
                    for k in range(tmp.shape[0]):
                        s = int(tmp[k].shape[0] / dpes.xx.shape[1])
                        dxxdlst.append(tmp[k].reshape(dpes.xx.shape[1], s))

                    dxxd[i][t] = np.array(dxxdlst)
                    if tmp_print:
                        print('unpack AEV derivative data')
                        tmp_print = False
                except:
                    pass
            else:
                if tmp_print:
                    print('no AEV derivative data was unpacked')
                    tmp_print = False
            xxd[i][dpes.num_nn] = dpes.yy[i].tolist()
        try:
            wd[i] = dpes.ww[i]
        except:
            pass
        if fdata:
            try:
                fd[i] = dpes.ff[i]
            except:
                pass

    xxp = [dpes.xp[dpes.xip[i]:dpes.xfp[i]] for i in range(dpes.ndat)]

    dpes.xdat = xxd
    dpes.pdat = xxp
    dpes.w = np.array(wd)
    if fdata:
        try:
            dpes.dxdat = dxxd
            #=====================================================
            tps = timeit.default_timer()
            d0 = len(dpes.dxdat)
            d1 = []
            d2 = []
            d3 = []
            d4 = []
            for i in range(len(dpes.dxdat)):
                d1.append(len(dpes.dxdat[i]))
                for j in range(len(dpes.dxdat[i])):
                    d2.append(len(dpes.dxdat[i][j]))
                    for k in range(len(dpes.dxdat[i][j])):
                        d3.append(len(dpes.dxdat[i][j][k]))
                        for v in range(len(dpes.dxdat[i][j][k])):
                            d4.append(len(dpes.dxdat[i][j][k][v]))

            dpes.padded_dxdat = np.zeros((d0, max(d1), max(d2), max(d3), max(d4)))
            for i in range(len(dpes.dxdat)):
                for j in range(len(dpes.dxdat[i])):
                    for k in range(len(dpes.dxdat[i][j])):
                        for v in range(len(dpes.dxdat[i][j][k])):
                            dpes.padded_dxdat[i][j][k][v][:len(dpes.dxdat[i][j][k][v])] = dpes.dxdat[i][j][k][v]

            tpd = timeit.default_timer()
            print('Time to fininsh padding dxdat: ', tpd-tps)
            #=====================================================

        except:
            pass

        try:
            dpes.fdat = np.array(fd)
            #=====================================================
            d0 = len(dpes.fdat)
            d1 = []
            d2 = []
            for i in range(len(dpes.fdat)):
                d1.append(len(dpes.fdat[i]))
                for j in range(len(dpes.fdat[i])):
                    d2.append(len(dpes.fdat[i][j]))
            dpes.fd2 = d2
            dpes.padded_fdat = np.zeros((d0, max(d1), max(d2)))
            for i in range(len(dpes.fdat)):
                for j in range(len(dpes.fdat[i])):
                    dpes.padded_fdat[i][j][:len(dpes.fdat[i][j])] = dpes.fdat[i][j]
            #=====================================================
            print('unpacked force data')
        except:
            pass
    else:
        print('no force data was unpacked')



    dpes.ntrdat = (dpes.tvtmsk == 1).sum()
    dpes.nvldat = (dpes.tvtmsk == 0).sum()
    dpes.ntsdat = (dpes.tvtmsk == -1).sum()

    dpes.dimdat = len(dpes.xdat[0][0][0])
    for t in range(dpes.num_nn):
        assert (dpes.dimdat == len(dpes.xdat[0][t][0]))
    print("unpack_data: xdat -> dimdat:", dpes.dimdat)



def read_pca_aev_db_hdf(dpes, fname, ni=0, nf=None):
    """
    Method to read the AEV data base for the Data_pes object from a hdf5 file
    """

    try:
        f = h5py.File(fname, 'r')
    except IOError:
        print("read_aev_db_hdf: Could not open file: ", fname)
    with f:
        print("read_aev_db_hdf: reading file:", fname)

        # report what groups are available in this file
        print("hdf file has the following groups:")
        for g in f.keys():
            print(g)

        # pick g as the 'Base_Group' group
        print("reading the group: 'Base_Group'")
        g = f['Base_Group']

        # report metadata for group g
        print("group has the following metadata:")
        for m in g.attrs.keys():
            print('{} => {}'.format(m, g.attrs[m]))

        # report what data-sets are available in group g
        print("group has the following contents:")
        for m in g.keys():
            print('{} => {}'.format(m, g[m]))

        # get the types from the metadata info
        # if the data object has already been set up with a particular atom_types then
        # assert that the hdf5 data corresponds to the same atom_types ... otherwise exit
        # if no data_types has already been set then set it up based on the atom_types in the file

        got_types = list(g.attrs['Types'])
        print("data file built for atom types:", got_types)
        if hasattr(dpes, 'atom_types'):
            print("Data object already has atom_types defined:", dpes.atom_types)
            print("Proceeding if consistent with the file types contents ... ", end='')
            assert (checkEqual(dpes.atom_types, got_types))
            print("yes, consistent.")
        else:
            print("Data object has no atom_types defined yet ... setting it per the file")
            dpes.initialize(got_types)

        print("reading data sets from hdf file ... ", end='')

        dpes.ndat = g.attrs['ndat']
        assert (ni >= 0 and ni < dpes.ndat)
        if nf != None:
            assert (nf > ni and nf <= dpes.ndat)
            dpes.ndat = nf
        dpes.ndat -= ni

        dpes.full_symb_data = [(s.strip('][').translate({ord(c): None for c in "'"}).split(', ')) for s in
                               g['symbols'][ni:nf].tolist()]
        dpes.meta = [tuple(m) for m in g['meta'][ni:nf].tolist()]
        dpes.xit = g['xit'][ni:nf]
        dpes.xft = g['xft'][ni:nf]
        dpes.yy = g['yy'][ni:nf]
        dpes.xip = g['xip'][ni:nf]
        dpes.xfp = g['xfp'][ni:nf]
        dpes.tvtmsk = g['tvtmsk'][ni:nf]
        if nf == None:
            xif = dpes.xfp[dpes.ndat - 1 - ni]
        else:
            xif = dpes.xfp[nf - 1 - ni]

        xi0 = 0
        if ni != 0:
            xi0 = dpes.xip[0]
            dpes.xip = [x - xi0 for x in dpes.xip]
            dpes.xfp = [x - xi0 for x in dpes.xfp]
            dpes.xit = [[xt - xi0 for xt in x] for x in dpes.xit]
            dpes.xft = [[xt - xi0 for xt in x] for x in dpes.xft]

        dpes.xx = g['xx'][xi0:xif]
        dpes.xp = g['xp'][xi0:xif]

    return




def read_aev_db_hdf(dpes, fname, ni=0, nf=None, newhdf=newhdfg):
    """
            Method to read the AEV data base for the Data_pes object from a hdf5 file
        """

    try:
        f = h5py.File(fname, 'r')
    except IOError:
        print("read_aev_db_hdf: Could not open file: ", fname)
        sys.exit()

    with f:
        print("read_aev_db_hdf: reading file:", fname)

        # report what groups are available in this file
        print("hdf file has the following groups:")
        for g in f.keys():
            print(g)

        # pick g as the 'Base_Group' group
        print("reading the group: 'Base_Group'")
        g = f['Base_Group']

        # report metadata for group g
        print("group has the following metadata:")
        for m in g.attrs.keys():
            print('{} => {}'.format(m, g.attrs[m]))

        # report what data-sets are available in group g
        print("group has the following contents:")
        for m in g.keys():
            print('{} => {}'.format(m, g[m]))

        # get the types from the metadata info
        # if the data object has already been set up with a particular atom_types then
        # assert that the hdf5 data corresponds to the same atom_types ... otherwise exit
        # if no data_types has already been set then set it up based on the atom_types in the file

        got_types = list(g.attrs['Types'])
        print("data file built for atom types:", got_types)
        if hasattr(dpes, 'atom_types'):
            print("Data object already has atom_types defined:", dpes.atom_types)
            print("Proceeding if consistent with the file types contents ... ", end='')
            assert (checkEqual(dpes.atom_types, got_types))
            print("yes, consistent.")
        else:
            print("Data object has no atom_types defined yet ... setting it per the file")
            dpes.initialize(got_types)

        print("reading data sets from hdf file ... ", end='')

        dpes.ndat = g.attrs['ndat']
        assert (ni >= 0 and ni < dpes.ndat)
        if nf != None:
            assert (nf > ni and nf <= dpes.ndat)
            dpes.ndat = nf
        dpes.ndat -= ni

        try:
            dpes.full_symb_data = [(s.strip('][').translate({ord(c): None for c in "'"}).split(', ')) for s in
                                   g['symbols'][ni:nf].tolist()]
        except:
            dpes.full_symb_data = g['symbols'][ni:nf].tolist()

        if isinstance(dpes.full_symb_data[0][0], (bytes, bytearray)):
            dpes.full_symb_data = [[s.decode("utf-8") for s in sa] for sa in dpes.full_symb_data]

        if newhdf:
            e2n     = lambda i : i or None      # convert empty string to None
            g_met_e = g['met_e'][ni:nf]
            if isinstance(g['met_s'][0][0], (bytes, bytearray)):
                g_met_o = [[e2n(gss.decode("utf-8")) for gss in gs] for gs in g['met_s']][ni:nf]
            else:
                g_met_o = [[e2n(gss) for gss in gs] for gs in g['met_s']][ni:nf]
            g_met_f = [gf.reshape(-1,3) for gf in g['met_f']][ni:nf]
            dpes.meta = [tuple([m[0]]+m[1]+[m[2]]) for m in zip(g_met_e,g_met_o,g_met_f)]
        else:
            dpes.meta = [tuple(m) for m in g['meta'][ni:nf].tolist()]
        dpes.xit = g['xit'][ni:nf]
        dpes.xft = g['xft'][ni:nf]
        dpes.yy = g['yy'][ni:nf]
        try:
            dpes.ww = g['ww'][ni:nf]
        except:
            print('no weight in data')
        dpes.xip = g['xip'][ni:nf]
        dpes.xfp = g['xfp'][ni:nf]
        try:
            dpes.ff = g['ff'][ni:nf]
            print('force date was read from hdf file')
        except:
            print('force data was not read from hdf file')
        dpes.tvtmsk = g['tvtmsk'][ni:nf]

        if nf == None:
            xif = dpes.xfp[dpes.ndat - 1 - ni]
        else:
            xif = dpes.xfp[nf - 1 - ni]

        xi0 = 0
        if ni != 0:
            xi0 = dpes.xip[0]
            dpes.xip = [x - xi0 for x in dpes.xip]
            dpes.xfp = [x - xi0 for x in dpes.xfp]
            dpes.xit = [[xt - xi0 for xt in x] for x in dpes.xit]
            dpes.xft = [[xt - xi0 for xt in x] for x in dpes.xft]

        dpes.xx = g['xx'][xi0:xif]
        dpes.xp = g['xp'][xi0:xif]
        tmp_print = True
        try:
            dpes.dxx = g['dxx'][xi0:xif]
            if tmp_print:
                print('AEV derivative was read from hdf file')
                tmp_print = False
        except:
            if tmp_print:
                print('AEV derivative was not read from hdf file')


    return



def write_pca_aev_db_hdf(dpes, fname, dpca):
    """
    Method to save the AEV PCA data base for the Data_pes object to a hdf5 file
    """
    print("write_pca_aev_db_hdf: writing file:", fname)

    with h5py.File(fname, 'w') as f:
        # create a group g entitled "Base_Group" for the data
        g = f.create_group('Base_Group')
        # create some metadata useful in each group
        metadata = {'Date': time.time(),
                    'User': 'HNN',
                    'Types': dpes.atom_types,
                    'ndat': dpes.ndat,
                    'OS': os.name}
        g.attrs.update(metadata)

        # create group "PCA" for PCA details employed to gen the above projected data
        g_pca = f.create_group('PCA')
        metadata = {'pevthr': dpca.pevthr,
                'npc': dpca.npc,
                    'ncomp': [dpca.ncomp[t] for t in range(dpes.num_nn)],
                    'natot': dpca.natot
                    }
        g_pca.attrs.update(metadata)

        # create a string data type for use in the hdf file encoding below for character/string data
        string_dt = h5py.special_dtype(vlen=str)

        # create/write datasets within group g
        g.create_dataset('symbols', data=np.array(dpes.full_symb_data, dtype=object), dtype=string_dt)
        g.create_dataset('meta', data=np.array(dpes.meta, dtype=object), dtype=string_dt)
        g.create_dataset('xit', data=dpes.xit)
        g.create_dataset('xft', data=dpes.xft)
        g.create_dataset('xx', data=dpes.xx)
        g.create_dataset('yy', data=dpes.yy)
        g.create_dataset('xip', data=dpes.xip)
        g.create_dataset('xfp', data=dpes.xfp)
        g.create_dataset('xp', data=dpes.xp)
        g.create_dataset('tvtmsk', data=dpes.tvtmsk)

        for t in range(dpes.num_nn):
            g_pca.create_dataset('mean' + str(t), data=dpca.Xpca[t].mean_)
            g_pca.create_dataset('components' + str(t), data=dpca.Xpca[t].components_)
            g_pca.create_dataset('explained_variance' + str(t), data=dpca.Xpca[t].explained_variance_)
    return



def write_aev_db_hdf(dpes, fname, newhdf=newhdfg):
    """
    Method to save the AEV data base for the Data_pes object to a hdf5 file
    """
    print("write_aev_db_hdf: writing file:", fname)

    with h5py.File(fname, 'w') as f:
        # create a group g entitled "Base_Group"
        # this is not necessary, but it can come in handy to have multiple groups within a file
        g = f.create_group('Base_Group')

        # create a string data type for use in the hdf file encoding below for character/string data
        string_dt = h5py.special_dtype(vlen=str)
        flt_dt = h5py.special_dtype(vlen=np.dtype('float64'))
        double_dt = h5py.vlen_dtype(np.dtype('float64'))

        # create/write datasets within group g

        #padd the species in dpes.full_symb_data to make them uniform length so they can be properly stored in hdf.
        #32 was chosen arbitraily as a molecule size larger than any we plan any working with in the near future. Can be changed if need be.
        full_symb_data_padded = []
        for i in range(len(dpes.full_symb_data)):
            full_symb_data_padded.append(dpes.full_symb_data[i] + [''] * (32 - len(dpes.full_symb_data[i])))

        g.create_dataset('symbols', data=np.array(full_symb_data_padded, dtype=object), dtype=string_dt)

        if newhdf:
            n2e     = lambda i : i or ''        # convert None to empty string
            met_e = np.array(dpes.meta, dtype=object)[:,0].astype('float64')                  # energy
            met_o = np.array(dpes.meta, dtype=object)[:,1:7]                                  # name, method, type, name, path, xyz
            met_s = np.array([[n2e(m) for m in ma] for ma in met_o], dtype=met_o.dtype)  # convert all None to '' in met_s
            met_f = np.array(dpes.meta, dtype=object)[:,7].astype(np.ndarray)                 # force
            g.create_dataset('met_e', data=met_e)
            g.create_dataset('met_s', data=met_s, dtype=string_dt)
            g.create_dataset('met_f', (met_f.shape[0],), dtype=double_dt)
            for i,met in enumerate(met_f):
                g['met_f'][i]=met.flatten()
        else:
            g.create_dataset('meta', data=np.array(dpes.meta, dtype=object), dtype=string_dt)
        g.create_dataset('xit', data=dpes.xit)
        g.create_dataset('xft', data=dpes.xft)
        g.create_dataset('xx', data=dpes.xx)
        try:
            g.create_dataset('dxx', (dpes.dxx.__len__(),), dtype=flt_dt)
            g['dxx'][:] = np.array(dpes.dxx, dtype=object)
            print('AEV derivative was written in hdf5')
        except:
            print('AEV derivative was not written in hdf5')
        try:
            g.create_dataset('ff', (dpes.ff.shape[0], 1), dtype=flt_dt)
            g['ff'][:] = np.array(dpes.ff)
            print('force data was written in hdf5')
        except:
            print('force data was not written in hdf5')
        g.create_dataset('yy', data=dpes.yy)
        g.create_dataset('ww', data=dpes.ww)
        g.create_dataset('xip', data=dpes.xip)
        g.create_dataset('xfp', data=dpes.xfp)
        g.create_dataset('xp', data=dpes.xp)
        g.create_dataset('tvtmsk', data=dpes.tvtmsk)

        # create some metadata useful in group g
        metadata = {'Date': time.time(),
                    'User': 'HNN',
                    'Types': dpes.atom_types,
                    'ndat': dpes.ndat,
                    'OS': os.name}
        g.attrs.update(metadata)

    return

