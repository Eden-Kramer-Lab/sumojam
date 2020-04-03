import sumojam.fastnum as _fm
import sumojam.hc_bcast as _hcb
import numpy as _N
from sumojam.tools.fitutil import  emMKPOS_sep1A, sepHash, colorclusters, contiguous_pack2
from posteriorUtil import MAPvalues2, gam_inv_gam_dist_ML
import clrs 
from filter import gauKer
import time as _tm
from EnDedirs import resFN, datFN
import matplotlib.pyplot as _plt
import scipy.stats as _ss
import openTets as _oT
import utilities as _U
import posteriorUtil as _pU
import pickle

twpi = 2*_N.pi
fourpi2 = 4*_N.pi*_N.pi
wdSpc = 1

def stochasticAssignment(epc, it, M, K, l0, f, q2, u, Sg, iSg, _f_u, _u_u, _f_q2, _u_Sg, nSpks, t0, mAS, xAS, rat, econt, max_so_far, cgz, qdrSPC, qdrMKS, exp_arg, freeClstr, clstsz):
    m1stSignalClstr = 0
    #  Msc   Msc signal clusters
    #  M     all clusters, including nz clstr.  M == Msc when not using nzclstr
    #  Gibbs sampling
    #  parameters l0, f, q2
    #  mASr, xASr   just the mark, position of spikes btwn t0 and t1
    #  qdrMKS   quadratic distance from all marks to the M cluster centers

    iq2        = 1./q2
    pkFR       = _N.log(l0) - 0.5*_N.log(twpi*q2)   #  M

    mkNrms = -0.5*_N.log(twpi*_N.linalg.det(Sg))
    rnds       = _N.random.rand(nSpks)

    _fm.cluster_probabilities(pkFR, mkNrms, exp_arg, econt, rat, rnds, f, xAS,\
                              iq2, qdrSPC, mAS, u, iSg, qdrMKS, max_so_far, M, \
                              nSpks, K)
    _fm.set_occ_cgz(clstsz, rat, rnds, cgz[it], M, nSpks)


def stochasticAssignment_2d(epc, it, M, K, l0, fx, fy, q2x, q2y, u, Sg, iSg, nSpks, t0, mAS, xAS, yAS, rat, econt, max_so_far, cgz, qdrSPC, qdrMKS, exp_arg, freeClstr, clstsz):
    m1stSignalClstr = 0
    #  Msc   Msc signal clusters
    #  M     all clusters, including nz clstr.  M == Msc when not using nzclstr
    #  Gibbs sampling
    #  parameters l0, fx     sizeM, q2
    #  mASr, xASr   just the mark, position of spikes btwn t0 and t1
    #  qdrMKS   quadratic distance from all marks to the M cluster centers

    iq2x        = 1./q2x
    iq2y        = 1./q2y

    pkFR       = _N.log(l0) - 0.5*_N.log(fourpi2*q2x*q2y)   #  M
    mkNrms = -0.5*_N.log(twpi*_N.linalg.det(Sg))

    rnds       = _N.random.rand(nSpks)

    _fm.cluster_probabilities_2d(pkFR, mkNrms, exp_arg, econt, rat, rnds, fx,
                                 fy, xAS, yAS, iq2x, iq2y, qdrSPC, mAS, u,
                                 iSg, qdrMKS, max_so_far, M, nSpks, K)
    _fm.set_occ_cgz(clstsz, rat, rnds, cgz[it], M, nSpks)    


    
