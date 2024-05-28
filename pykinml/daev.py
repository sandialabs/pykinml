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

import timeit
import numpy as np
import torch





#====================================================================================================
def cal_hessian_dl(dpes, E):
	hes = [[] for b in range(E.shape[0])]
	for i in range(E.shape[0]):
		y = E[i]
		# y.backward(retain_graph=True)

		a = torch.autograd.grad(outputs=y, inputs=dpes.xb[0], create_graph=True)
		aC = a[0].reshape(-1)
		# aH = a[1].reshape(-1)
		hes_C = []
		for jj in range(aC.shape[0]):
			tmp_C, = torch.autograd.grad(outputs=aC[jj], inputs=dpes.xb[0], retain_graph=True)
			hes_C.append(tmp_C)
		# hes_H = []
		# for jj in range(aH.shape[0]):
		# 	tmp_H, = torch.autograd.grad(outputs=aH[jj], inputs=dpes.xb[0][1], retain_graph=True)
		# 	hes_H.append(tmp_H)
		H_C = torch.stack(hes_C).reshape(a[0].shape + tmp_C.shape)
		# H_H = torch.stack(hes_H).reshape(a[1].shape + tmp_H.shape)
		hes[i] = H_C
	return H_C

def cal_dEdxyz_dl(dpes, E):
        dEdxyz = [[] for b in range(E.shape[0])]
        for i in range(dpes.ndat):
                y = E[i]
                a = torch.autograd.grad(outputs=y, inputs=dpes.xb[0], create_graph=True, allow_unused=True)
                #a = torch.autograd.grad(outputs=y, inputs=dpes.xb[0], create_graph=True, allow_unused=False)
                dE = torch.zeros((3 * dpes.full_symb_data[i].__len__()), device=dpes.device)
                for j in torch.nonzero(sum([dpes.inddl[0][0] == i]), as_tuple=True)[0]:
                        dE = dE + torch.matmul(a[0][j], torch.tensor(dpes.dxb[0][0][j], device=dpes.device))
                for j in torch.nonzero(sum([dpes.inddl[0][1] == i]), as_tuple=True)[0]:
                        dE = dE + torch.matmul(a[1][j], torch.tensor(dpes.dxb[0][1][j], device=dpes.device))

                dEdxyz[i] = dE
        return dEdxyz



def cal_dEdxyz_tr(dpes, E, b, bpath):
        dEdxyz = [[] for i in range(E.shape[0])]
        dEdxyz_true = [[] for i in range(E.shape[0])]
        dxtr = torch.load(bpath+'dxtr_batch_'+str(b), map_location=dpes.device)
        ftr = torch.load(bpath+'ftr_batch_'+str(b), map_location=dpes.device)
        t0 = timeit.default_timer()
        fa = torch.autograd.grad(outputs=E.sum(), inputs=dpes.xbtr[b], create_graph=True)
        for i in range(E.shape[0]):
                t1 = timeit.default_timer()
                dE = torch.zeros((len(dxtr[0][0][0])), device=dpes.device)
                #dE = torch.zeros((len(dpes.dxbtr[b][0][0][0])), device=dpes.device)
                for j in torch.nonzero(sum([dpes. indtr[b][0] == i]), as_tuple=True)[0]:
                        dE = dE + torch.matmul(fa[0][j], dxtr[0][j].to(dpes.device))
                for j in torch.nonzero(sum([dpes. indtr[b][1] == i]), as_tuple=True)[0]:
                        dE = dE + torch.matmul(fa[1][j], dxtr[1][j].to(dpes.device))

                t2 = timeit.default_timer()
                dEdxyz[i] = dE#[0:dpes.pfdims_tr[b][0][i]]
                dEdxyz_true[i] = ftr[i]#[0:dpes.pfdims_tr[b][0][i]]#dpes.fbtr[b][i][0:dpes.pfdims_tr[b][0][i]]
                #dEdxyz_true[i] = dpes.fbtr[b][i][0:dpes.pfdims_tr[b][0][i]]
                dpes.tf[0] += t1 - t0
                dpes.tf[1] += t2 - t1
        return dEdxyz, dEdxyz_true

def cal_dEdxyz_tr_lf(dpes, E, b):
        dEdxyz = [[] for i in range(E.shape[0])]
        dEdxyz_true = [[] for i in range(E.shape[0])]
        t0 = timeit.default_timer()
        fa = torch.autograd.grad(outputs=E.sum(), inputs=dpes.xbtr_lf[b], create_graph=True)
        for i in range(E.shape[0]):
                t1 = timeit.default_timer()
                dE = torch.zeros((len(dpes.dxbtr_lf[b][0][0][0])), device=dpes.device)
                for j in torch.nonzero(sum([dpes. indtr_lf[b][0] == i]), as_tuple=True)[0]:
                        dE = dE + torch.matmul(fa[0][j], dpes.dxbtr_lf[b][0][j])
                for j in torch.nonzero(sum([dpes. indtr_lf[b][1] == i]), as_tuple=True)[0]:
                        dE = dE + torch.matmul(fa[1][j], dpes.dxbtr_lf[b][1][j])

                t2 = timeit.default_timer()
                dEdxyz[i] = dE
                dEdxyz_true[i] = dpes.fbtr_lf[b][i][0:dpes.pfdims_tr[b][0][i]]

                dpes.tf[0] += t1 - t0
                dpes.tf[1] += t2 - t1
        return dEdxyz, dEdxyz_true


def cal_dEdxyz_ts(dpes, E, b, bpath):
        dEdxyz = [[] for i in range(E.shape[0])]
        dEdxyz_true = [[] for i in range(E.shape[0])]
        dxts = torch.load(bpath+'dxts_batch_'+str(b), map_location=dpes.device)
        fts = torch.load(bpath+'fts_batch_'+str(b), map_location=dpes.device)
        fa = torch.autograd.grad(outputs=E.sum(), inputs=dpes.xbts[b], create_graph=True)
        for i in range(E.shape[0]):
                dE = torch.zeros((len(dxts[0][0][0])), device=dpes.device)
                for j in torch.nonzero(sum([dpes.indts[b][0] == i]), as_tuple=True)[0]:
                        dE = dE + torch.matmul(fa[0][j], dxts[0][j])
                for j in torch.nonzero(sum([dpes.indts[b][1] == i]), as_tuple=True)[0]:
                        dE = dE + torch.matmul(fa[1][j], dxts[1][j]) 
                dEdxyz[i] = dE[0:dpes.pfdims_ts[b][0][i]]
                dEdxyz_true[i] = fts[i][0:dpes.pfdims_ts[b][0][i]]
        return dEdxyz, dEdxyz_true

def cal_dEdxyz_ts_lf(dpes, E, b):
        dEdxyz = [[] for b in range(E.shape[0])]
        dEdxyz_true = [[] for b in range(E.shape[0])]
        fa = torch.autograd.grad(outputs=E.sum(), inputs=dpes.xbts_lf[b], create_graph=True)
        for i in range(E.shape[0]):
                dE = torch.zeros((len(dpes.dxbts_lf[b][0][0][0])), device=dpes.device)
                for j in torch.nonzero(sum([dpes.indts_lf[b][0] == i]), as_tuple=True)[0]:
                        dE = dE + torch.matmul(fa[0][j], dpes.dxbts_lf[b][0][j])
                for j in torch.nonzero(sum([dpes.indts_lf[b][1] == i]), as_tuple=True)[0]:
                        dE = dE + torch.matmul(fa[1][j], dpes.dxbts_lf[b][1][j])
                dEdxyz[i] = dE[0:dpes.pfdims_ts[b][0][i]]
                dEdxyz_true[i] = dpes.fbts_lf[b][i][0:dpes.pfdims_ts[b][0][i]]
        return dEdxyz, dEdxyz_true


def cal_dEdxyz_vl(dpes, E, b, bpath):
        dEdxyz = [[] for i in range(E.shape[0])]
        dEdxyz_true = [[] for i in range(E.shape[0])]
        dxvl = torch.load(bpath+'dxvl_batch_'+str(b), map_location=dpes.device)
        fvl = torch.load(bpath+'fvl_batch_'+str(b), map_location=dpes.device)
        fa = torch.autograd.grad(outputs=E.sum(), inputs=dpes.xbvl[b], create_graph=True)
        for i in range(E.shape[0]):
                dE = torch.zeros((len(dxvl[0][0][0])), device=dpes.device)
                for j in torch.nonzero(sum([dpes.indvl[b][0] == i]), as_tuple=True)[0]:
                        dE = dE + torch.matmul(fa[0][j], dxvl[0][j])
                for j in torch.nonzero(sum([dpes.indvl[b][1] == i]), as_tuple=True)[0]:
                        dE = dE + torch.matmul(fa[1][j], dxvl[1][j])

                dEdxyz[i] = dE[0:dpes.pfdims_vl[b][0][i]]
                dEdxyz_true[i] = fvl[i][0:dpes.pfdims_vl[b][0][i]]
        return dEdxyz, dEdxyz_true

def cal_dEdxyz_vl_lf(dpes, E, b):
        dEdxyz = [[] for b in range(E.shape[0])]
        dEdxyz_true = [[] for b in range(E.shape[0])]
        fa = torch.autograd.grad(outputs=E.sum(), inputs=dpes.xbvl_lf[b], create_graph=True)
        for i in range(E.shape[0]):
                dE = torch.zeros((len(dpes.dxbvl_lf[b][0][0][0])), device=dpes.device)
                for j in torch.nonzero(sum([dpes.indvl_lf[b][0] == i]), as_tuple=True)[0]:
                        dE = dE + torch.matmul(fa[0][j], dpes.dxbvl_lf[b][0][j])
                for j in torch.nonzero(sum([dpes.indvl_lf[b][1] == i]), as_tuple=True)[0]:
                        dE = dE + torch.matmul(fa[1][j], dpes.dxbvl_lf[b][1][j])

                dEdxyz[i] = dE[0:dpes.pfdims_vl[b][0][i]]
                dEdxyz_true[i] = dpes.fbvl_lf[b][i][0:dpes.pfdims_vl[b][0][i]]
        return dEdxyz, dEdxyz_true




