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
import numpy as np
import matplotlib.pyplot as plt

from ase import Atoms

from pykinml import data

verbose = False
diagnose = False

#====================================================================================================
class Aev():
	def __init__(self, atom_types,nrho_rad=32,nrho_ang=8,nalpha=8):
		"""
		atom_types: list of strings
			n long list of strings
			[ type_1, type_2, ... type_n ]
			e.g. ["H", "O", "C"]
		rad_par: 2D array of doubles (nr x 2)
			[
				[etar_1, rhor_1],
				[etar_2, rhor_2],
							...
				[eta_nr, rho_nr]
			]
		ang_par: 2D array of doubles (na x 4)
			[
				[etaa_1, rhoa_1, zeta_1, theta_1],
				[etaa_2, rhoa_2, zeta_2, theta_2],
							...
				[etaa_na, rhoa_na, zeta_na, theta_na]
			]
		R_c: 1D vector of two doubles
			R_c[0]: radial-SF cutoff radius
			R_c[1]: angular-SF cutoff radius

		conf: 1D tensor of strings
			this is a configuration ... an N long list of atom names
			e.g. [ "H", C", "O", "O", "H"]
		"""

		# R_c[0]: radial SF cutoff radius (Angstrom)
		# R_c[1]: angular SF cutoff radius (Angstrom)
		R_c = np.array([4.6, 3.1])

		# spec for radial SFs
		#nrho_rad  = 2 # 32             		# number of radial rho values in radial SF
		drho_rad  = R_c[0]/nrho_rad
		delta_rad = (2./3.)*drho_rad
		eta_rad   = 1.0/delta_rad**2    # a single eta
		rhov_rad  = np.linspace(drho_rad/2.0,R_c[0]-drho_rad/2.0,nrho_rad)
		etav_rad  = np.array([eta_rad])
		rad_par   = two_d_2col_pack(etav_rad,rhov_rad)

		# spec for angular SFs
		drho_ang  = R_c[1]/nrho_ang
		delta_ang = (2./3.)*drho_ang
		eta_ang   = 1.0/delta_ang**2
		rhov_ang  = np.linspace(drho_ang/2.0,R_c[1]-drho_ang/2.0,nrho_ang)
		etav_ang  = np.array([eta_ang])

		alphav    = np.linspace(0.0,math.pi,nalpha)
		zeta      = 8.0
		zetav     = np.array([zeta])

		ang_par   = two_d_4col_pack(etav_ang, rhov_ang, zetav, alphav)

		if verbose:
			print("radial AEV SF params:\neta, rho:\n",rad_par,sep="")
			print("radial cutoff radius: R_c:",R_c[0])
			print("angular AEV SF params:\neta, rho, zeta, alpha:\n",ang_par,sep="")
			print("angular cutoff radius: R_c",R_c[1])

		self.rad_par = rad_par
		self.ang_par = ang_par
		self.types   = atom_types
		self.R_c     = R_c
		nr = rad_par.shape[0]
		na = ang_par.shape[0]
		n  = len(atom_types)
		self.n_rad   = n
		self.n_ang   = round(n*(n+1)/2)
		self.dout    = nr * self.n_rad + na * self.n_ang
		self.rad_typ = atom_types.copy()
		self.ang_typ = [np.array([atom_types[i],atom_types[j]]) for i in range(0,n) for j in range(i,n)]
		self.tag     = [" "] * self.dout

		"""
		n is the number of atom types
		nr is the number of radial SFs (# of pairs of parameters in rad_par)
		na is the number of angular SFs (# of quad-tuples of parameters in ang_par)
		self.n_rad is the number of radial blocks in the AEV, each block corresponding to an atom type
				   and each block being composed of nr elements
		self.n_ang is the number of angular blocks in the AEV, each block corresponding to a pair of atom types
			       irrespective of ordering, and each block being composed of na elements
		self.dout is the full size of the AEV
		self.rad_typ is the list of atom types in the radial AEV components, a direct copy of atom_types
		self.ang_typ is the list of atom type pairs in the angular AEV components
		self.tag is primarily for diagnostic purposes
			* a tag of "H0" indicates that the corresponding element of the AEV is a radial element
			(because only one atom type is included in the tag), that it's the radial element for atom type H
			and that it's the 0-th element in the radial H block. We would have elements H0, H1, ..., H<nr-1>
			* a tag of "HC0" indicates that the corresponding element of the AEV is an angular element
			(because a pair of atom types is included in the tag), that it's the angular element for atom types H and C in any order
			and that it's the 0-th element in the angular block. We would have elemtns HC0, HC1, ..., HC<na-1>
		"""
		for kk in range(0,self.n_rad):	# loop over radial AEV components
			kknr = kk*nr
			for l in range (0,nr):
				self.tag[kknr+l] = atom_types[kk]+str(l)

		q_offset = nr * self.n_rad
		for kk in range(0,self.n_ang):	# loop over angular AEV components
			kkna = kk*na
			for l in range (0,na):
				self.tag[q_offset+kkna+l] = self.ang_typ[kk][0]+self.ang_typ[kk][1]+str(l)

	def bld_index_sets(self,conf):

		#builds index sets for a given configuration

		assert all(elt in self.types for elt in conf), "Error: one elt in conf is not in atom_types"

		N = len(conf)

		# build radial index sets S_\tau
		rad_ind_set=[np.empty(0,int)] * self.n_rad

		for i in range(0,N):  # loop over atoms in conf
			for j in range (0,self.n_rad):  # loop over elements of index set
				if self.rad_typ[j] == conf[i]:
					rad_ind_set[j]=np.append(rad_ind_set[j],np.array([i]))

		# build angular index sets S_{\tau,\kappa}
		ang_ind_set      = [np.empty((0,2),int) for i in range(0,self.n_rad) for j in range(i,self.n_rad)]

		for i in range(0,N-1):
			for j in range(i+1,N):
				for k in range(0,self.n_ang):
					tp1 = self.ang_typ[k][0]
					tp2 = self.ang_typ[k][1]
					if ( (conf[i] == tp1 and conf[j] == tp2) or
						 (conf[i] == tp2 and conf[j] == tp1)	):
						ang_ind_set[k]=np.append(ang_ind_set[k],np.array([[i,j]],int),axis=0)

		return rad_ind_set, ang_ind_set

	def eval (self, conf, rad_ind_set, ang_ind_set, xmat):
		"""
        Inputs
        ------
        xmat : 2D npt x din numpy array of doubles
            npt x din array of npt points in din dimensions
            each of the npt rows is a 3N long vector x
            where x is the 1D array composed of the coordinates of all the N atoms in a conf
          [
           [x_{1,1} , x_{1,2} , ... x_{1,din} ],
           [x_{2,1} , x_{2,2} , ... x_{2,din} ],
               ...
           [x_{npt,1}, x_{npt,2}, ... x_{npt,din}]
          ]

        Returns
        -------
        numpy array, 2d
          [
           [y_{1,1} , y_{1,2} , ... y_{1,dout} ],
           [y_{2,1} , y_{2,2} , ... y_{2,dout} ],
               ...
           [y_{npt,1}, y_{npt,2}, ... y_{npt,dout}]
          ]
      each row is the AEV of the specified geometry of the conf specified in x
		"""

		N = len(conf)  # number of atoms in configuration
		# the vector x that holds the coordinates of all atoms in the configuration
		# will be 3N long
		# [ x_11, x_12, x_13, x_21, x_22, x_23, ... x_N1, x_N2, x_N3 ]
		# where (x_i1, x_i2, x_i3) are the (x,y,z) coordinates of atom i in the config

		npt = xmat.shape[0]  # number of data points, each being 3N long x vector

		din = 3*N    # full dimension of x row
		y = np.zeros((npt,N,self.dout))
		n = len(self.types)

		for p in range(0,npt):     				# loop over data points
			if diagnose:
				print("=====================================================")
				print("Data point p:",p)
			x = np.reshape(xmat[p],(-1,3))
			for i in range(0,N):				# loop over atoms in configuration
				atom=conf[i]
				if diagnose:
					print("=====================================================")
					print("centered on atom # i:",i,", atom:", atom)
				#deal with radial SFs
				nst = len(rad_ind_set)
				nsf = self.rad_par.shape[0]		# nsf is # radial SFs
				if diagnose:
					print("Evaluating",nsf,"Radial Symmetry Functions")
					print("Looping over",nst,"unitary index sets, one for each atom type")
				for kk in range(0,nst):	# loop over unitary index sets
					elmt=self.rad_typ[kk]
					indset=rad_ind_set[kk]
					kknsf = kk*nsf
					if diagnose:
						print ("Type: ",elmt,": Index Set:",indset)
						print ("Starting index in radial block:",kknsf)
					for j in indset:		# loop over indices in index set if any
						if j != i:			# ignore case when j=i
							if diagnose:
								print ("working on atom # j:",j)
							Rij=np.linalg.norm(x[j]-x[i])    # ||x_j-x_i||
							for l in range(0,nsf):			 # loop over radial SFs
								q = kknsf + l
								par=self.rad_par[l]
								y[p][i][q] = y[p][i][q] + rad_kern(par,Rij,self.R_c[0])
								if diagnose:
									print("p:",p,", i:",i,", atom:",atom,", type:",elmt,", j:",j,", q:",q,", y:",y[p][i][q])
				q_offset = nsf * n

				#deal with angular SFs
				nst = len(ang_ind_set)
				nsf = self.ang_par.shape[0]		# nsf is # angular SFs
				if diagnose:
					print("Evaluating",nsf,"Angular Symmetry Functions")
					print("Looping over",nst,"pairwise index sets, one for each unordered pair of atom types")
				for kk in range(0,nst):	# loop over pairwise index sets
					elmt1=self.ang_typ[kk][0]		# type 1
					elmt2=self.ang_typ[kk][1]		# type 2
					indset=ang_ind_set[kk]				# index set
					kknsf = kk*nsf
					if diagnose:
						print ("Unordered pair of types:",elmt1,elmt2,": Index Set:",indset.tolist())
						print ("Starting index in angular block:",kknsf)
					for pair in indset:					# loop over pairwise index set
						j=pair[0]
						k=pair[1]
						if j != i and k != i:			# ignore when either j=i or k=i
							if diagnose:
								print ("Working on atom #s j:",j,", k:",k)
							vij=np.subtract(x[j],x[i])	# x_j - x_i
							vik=np.subtract(x[k],x[i])	# x_k - x_i
							Rij=np.linalg.norm(vij)		# ||x_j-x_i||
							Rik=np.linalg.norm(vik)		# ||x_k-x_i||
							theta=np.arccos( np.clip( 0.95*np.dot(vij,vik)/(Rij*Rik), -1.0, 1.0 ) )   # angle theta
							for l in range(0,nsf):		# loop over angular SFs
								q = q_offset + kknsf + l
								par=self.ang_par[l]
								y[p][i][q] = y[p][i][q] + ang_kern(par,Rij,Rik,theta,self.R_c[1])
								if diagnose:
									print("p:",p,", i:",i,", atom:",atom,", pair:",elmt1,elmt2,", j:",j,", k:",k,", q:",q,", y:",y[p][i][q])

		if (verbose):
			with np.printoptions(precision=6, suppress=True):
				print("=====================================================")
				print("AEV length:",self.dout)
				print("types:",self.types)
				print("conf:",conf)
				print("Radial Index sets:")
				for i in range (0,len(self.types)):
					print("Set(",self.rad_typ[i],") :",rad_ind_set[i])
				print("Angular Index sets:")
				for i in range (0,len(ang_ind_set)):
					print("Set(",self.ang_typ[i][0],",",self.ang_typ[i][1],") :",ang_ind_set[i].tolist())
				print("AEV tag:\n" '[%s]' % ' '.join(map(str, self.tag)))
				for p in range(0,npt):
					print("x:",x.tolist())
					for i in range(0,N):
						print("point:",p,", atom index i:",i,", atom:", conf[i],", AEV:")
						[print(self.tag[k],":",y[p][i][k]) for k in range(0,self.dout)]
				print("=====================================================")

		return y

#====================================================================================================
# kernel functions for AEV

# cutoff function
def fc(Rij,Rc):
	if Rij <= Rc:
		fcv = 0.5 * math.cos (math.pi * Rij/Rc) + 0.5
	else:
		fcv = 0.0
	return fcv

# kernel for Radial SFs
def rad_kern(par, Rij, Rc):
	eta = par[0]
	rho = par[1]
	rk  = math.exp( - eta*(Rij-rho)**2 ) * fc(Rij,Rc)
	return rk

# kernel for Angular SFs
def ang_kern(par,Rij,Rik,theta,Rc):
	eta   = par[0]
	rho   = par[1]
	zeta  = par[2]
	alpha = par[3]
	ra    = 0.5 * (Rij+Rik)
	rk    = math.exp ( - eta*(ra-rho)**2 ) * fc(Rij,Rc) * fc(Rik,Rc)
	ak    = pow(0.5+0.5*math.cos(theta-alpha),zeta) * rk
	return ak


#====================================================================================================
def two_d_2col_pack(a,b):
	"""
	given two 1d arrays, a and b, where a has na elements and b has nb elements
	return a 2d array that is (na.nb x 2) being the tensor product of the two 1d arrays
	thus, e.g. say na=2, nb=3, then the returned 6x2 2d array "pack" has the structure:
	a0 b0
	a0 b1
	a0 b2
	a1 b0
	a1 b1
	a1 b2
	"""
	pack=[]
	for xa in a:
		for xb in b:
			pack.append([xa, xb])
	pack=np.asarray(pack)
	return pack

#====================================================================================================
def two_d_4col_pack(a,b,c,d):
	"""
	given 4 1d arrays, a, b, c, d, being na, nb, nc, nd long
	return a 2d array that is (na.nb.nc.nd x 2), thus, say na=2, nb=1, nc=3, nd=2, get:
	a0 b0 c0 d0
	a0 b0 c0 d1
	a0 b0 c1 d0
	a0 b0 c0 d1
	a0 b0 c2 d0
	a0 b0 c2 d1
	a1 b0 c0 d0
	a1 b0 c0 d1
	a1 b0 c1 d0
	a1 b0 c0 d1
	a1 b0 c2 d0
	a1 b0 c2 d1
	"""
	pack=[]
	for xa in a:
		for xb in b:
			for xc in c:
				for xd in d:
					pack.append([xa, xb, xc, xd])
	pack=np.asarray(pack)
	return pack

#====================================================================================================


if __name__== "__main__":
	main()

