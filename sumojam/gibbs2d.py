"""
V1.2   use adaptive range for integrating over f
variance   0.0001 
"""
import stats_util as s_u
import scipy.stats as _ss
import os
import time as _tm
import py_cdf_smp as _pcs
import numpy as _N
import matplotlib.pyplot as _plt
from sumojam.tools.SUMOdirs import resFN, datFN
import pickle

import sumojam.posteriorUtil as _pU
import sumojam.gibbs1d_util as gAMxMu
#import sumojam.stochasticAssignment as _sA
import sumojam.stochasticAssignment as _sA
#import cdf_smp_tbl as _cdfs
import sumojam.cdf_smp_2d as _cdfs2dA
#import sumojam.ig_from_cdf_pkg as _ifcp
import sumojam.fastnum as _fm
import sumojam.tools.compress_gz_pyx as c_gz
import sumojam.iwish as _iw
from filter import gauKer
import sumojam.tools.anocc as _aoc2
#import cdf_smp_tbl as _cdfs


#  rotate dat when we get it

#import conv_px_tbl as _cpt

class MarkAndRF:
    ky_p_l0 = 0;    ky_p_fx  = 1;    ky_p_fy  = 2;    ky_p_q2x = 3
    ky_p_q2y = 4; 
    ky_h_l0_a = 0;  ky_h_l0_B=1;
    ky_h_f_u  = 2;  ky_h_f_q2=3;
    ky_h_q2_a = 4;  ky_h_q2_B=5;

    ky_h_l0_a = 0;  ky_h_l0_B=1;
    ky_h_f_u  = 2;  ky_h_f_q2=3;
    ky_h_q2_a = 4;  ky_h_q2_B=5;

    ky_p_u = 0;       ky_p_Sg = 1;
    ky_h_u_u = 0;     ky_h_u_Sg=1;
    ky_h_Sg_nu = 2;   ky_h_Sg_PSI=3;
    #  posIndx 0, spk01Indx  mksIndx

    dt      = 0.001
    occ_bin = 5
    #  position dependent firing rate
    ######################################  PRIORS
    twpi = 2*_N.pi
    #NExtrClstr = 5
    NExtrClstr = 3
    earliest   = 20000      #   min # of gibbs samples

    #  sizes of arrays
    NposHistBins = 200      #   # points to sample position with  (uniform lam(x)p(x))
    
    intvs = None    #
    intvs = None    #      
    dat   = None
    fkpth_prms = None

    resetClus = True

    diffPerMin = 1.  #  diffusion per minute
    epochs   = None
    adapt    = False
    use_conv_tabl = True

    outdir   = None
    polyFit  = True

    Nupx      = 200

    #  l0, q2      Sig    f, u
    t_hlf_l0 = int(1000*60*5)   # 10minutes
    t_hlf_q2 = int(1000*60*5)   # 10minutes

    diffusePerMin = 0.05    #  diffusion of certainty

    nz_q2               = 500
    nz_f                = 0

    Bx                  = 0    #  noise in xpos

    #  px and spatial integration limits.  Don't make larger accessible space
    xLo      = -6;    xHi      = 6   

    #  limits of conditional probability sampling of f and q2
    f_L   = -12;     f_H = 12   
    q2_L = 1e-6;    q2_H = 1e4

    oneCluster = False

    #q2_lvls    = _N.array([0.02**2, 0.05**2, 0.1**2, 0.2**2, 0.5**2, 1**2, 6**2, 100000**2])
    #Nupx_lvls  = _N.array([1000, 600, 200, 100, 60, 20, 12, 8])

    priors     = None
    tetr       = None    
    __JIFIResultDir__ = None


    def __init__(self, outdir, fn, intvfn, xLo=0, xHi=3, yLo=0, yHi=3, seed=1041, adapt=True, oneCluster=False, fkpth_prm_fn=None, fkpth_wgt=1, rotate_spc=0, __JIFIResultDir__=None, tet=None, seg_filtered_intvs=None):
        oo     = self
        oo.__JIFIResultDir__=__JIFIResultDir__
        oo.oneCluster = oneCluster
        oo.tetr = -1 if tet is None else tet        
        oo.adapt = adapt
        _N.random.seed(seed)

        ######################################  DATA input, define intervals
        # bFN = fn[0:-4]
        oo.outdir = outdir

        # if not os.access(bFN, os.F_OK):
        #     os.mkdir(bFN)

        if not os.access("%s.dat" % datFN(fn, create=False), os.F_OK):
            print("%s.dat" % datFN(fn, create=False))
            return 

        _dat    = _N.loadtxt("%s.dat" % datFN(fn, create=False))
        #oo.fkpth_dat    = _N.loadtxt("/Users/arai/usb/nctc/Workspace/EnDe/DATA/bond_md2d1004_fkpth.dat")
        if fkpth_prm_fn is not None:
            oo.fkpth_prms    = _N.loadtxt("%s.prms" % datFN(fkpth_prm_fn, create=False))
            oo.fkpth_wgt     = fkpth_wgt
            print("loaded fake path prms")

        mving_ts = _N.where(_dat[:, 8] == 1)[0]
        oo.dat  = _N.array(_dat[mving_ts])
        #  cut hash
        # if cuthash:
        #     _mks.cut_hash(oo.dat, thresh=100, max2nd=120)
        
        #oo.segs = segs

        #oo.datprms= _N.loadtxt("%s_prms.dat" % datFN(fn, create=False))

        intvs     = _N.loadtxt("%s.dat" % datFN(intvfn, create=False))
        _intvs  = _N.array(intvs*_dat.shape[0], dtype=_N.int) if seg_filtered_intvs is None else seg_filtered_intvs

        oo.intvs  = _N.array(_intvs)  # intervals in time of movement filtered data

        t0 = 0
        for i in range(len(intvs)-1):
            t1 = t0 + _N.sum(_dat[_intvs[i]:_intvs[i+1], 8])   #  only moving 
            print("%(0)d  %(1)d" % {"0" : t0, "1" : t1})
            oo.intvs[i] = t0
            oo.intvs[i+1] = t1
            t0 = t1
                
        oo.epochs    = oo.intvs.shape[0] - 1
        
        NT     = oo.dat.shape[0]
        oo.xLo = xLo   #  this limit used for spatial path sum
        oo.xHi = xHi   #  this limit used for spatial path sum
        oo.yLo = yLo   #  this limit used for spatial path sum
        oo.yHi = yHi   #  this limit used for spatial path sum

    def setup_spatial_sum_params(self, q2x=None, fx=None): 
        """
        *q_mlt_steps=2  means spatial bins [0.01, 0.02, 0.04, 0.08]...
        *if my q2 is bigger than max level, i'll just end up using more bins than neccessary. no prob 
        *bins_per_sd=5   heuristically, this is a good setting. 
        *q2x, fx     grid  
        """ 
        oo = self
        oo.q2_L, oo.q2_H = q2x   #  grid lims for calc of conditional posterior
        oo.f_L,  oo.f_H  = fx

    def gibbs(self, ITERS, K, priors, ep1=0, ep2=None, saveSamps=True, saveOcc=True, nz_pth=0., f_STEPS=13, q2_STEPS=13, f_SMALL=10, q2_SMALL=10, f_cldz=10, q2_cldz=10, minSmps=20, earliest=20000, cmprs=1, occ_bin=5, global_stop_condition=False):
        """ 
        gtdiffusion:  use ground truth center of place field in calculating variance of center.  Meaning of diffPerMin different 
        """ 
        oo = self
        print("RUNNING GIBBS GIBBS 2d  %(tet)d   %(save)s" % {"tet" : oo.tetr, "save" : oo.outdir})
        
        oo.earliest=earliest
        twpi     = 2*_N.pi
        pcklme   = {}

        oo.priors = priors
        oo.occ_bin = occ_bin

        ep2 = oo.epochs if (ep2 == None) else ep2
        oo.epochs = ep2-ep1

        ######################################  GRID for calculating
        ####  #  points in sum.  
        ####  #  points in uniform sampling of exp(x)p(x)   (non-spike interals)
        ####  #  points in sampling of f  for conditional posterior distribution
        ####  #  points in sampling of q2 for conditional posterior distribution
        ####  NSexp, Nupx, fss, q2ss

        #  numerical grid
        #ux = _N.linspace(oo.xLo, oo.xHi, oo.Nupx, endpoint=False)   # grid over which spatial integrals are calculated
        #uxr = ux.reshape((1, oo.Nupx))

        #  grid over which conditional probability of q2 adaptively sampled
        #  grid over which conditional probability of q2 adaptively sampled

        freeClstr = None
        if nz_pth > 0:
            oo.dat[:, 0] += nz_pth*_N.random.randn(oo.dat.shape[0])
            oo.dat[:, 1] += nz_pth*_N.random.randn(oo.dat.shape[0])
        xy      = oo.dat[:, 0:2]# + 0.01*_N.random.randn(oo.dat.shape[0], 2)

        init_mks    = _N.array(oo.dat[:, 3:3+K])  #  init using non-rot
        fit_mks    = init_mks

        f_q2_rate = (oo.diffusePerMin**2)/60000.  #  unit of minutes  
        
        ######################################  PRECOMPUTED

        #_ifcp.init()
        tau_l0 = oo.t_hlf_l0/_N.log(2)
        tau_q2 = oo.t_hlf_q2/_N.log(2)

        _cdfs2dA.init(oo.dt, oo.f_L, oo.f_H, oo.q2_L, oo.q2_H, f_STEPS, q2_STEPS, f_SMALL, q2_SMALL, f_cldz, q2_cldz, minSmps)

        M_max   = 50   #  100 clusters, max
        M_use    = 0     #  number of non-free + 5 free clusters

        
        for epc in range(ep1, ep2):
            print("^^^^^^^^^^^^^^^^^^^^^^^^    epoch %d" % epc)

            t0 = oo.intvs[epc]
            t1 = oo.intvs[epc+1]

            if epc > 0:
                tm1= oo.intvs[epc-1]
                #  0 10 30     20 - 5  = 15    0.5*((10+30) - (10+0)) = 15
                DT = t0-tm1

            #  _N.sum(px)*(xbns[1]-xbns[0]) = 1

            # ##  smooth the positions
            # smthd_pos = x[t0:t1] + oo.Bx*_N.random.randn(t1-t0)
            # ltL = _N.where(smthd_pos < oo.xLo)[0]
            # smthd_pos[ltL] += 2*(oo.xLo - smthd_pos[ltL])
            # gtR = _N.where(smthd_pos > oo.xHi)[0]
            # smthd_pos[gtR] += 2*(oo.xHi - smthd_pos[gtR])

            totalpcs, ap_mns, ap_sd2s, _ap_Ns = _aoc2.approx_maze_sungod(oo.dat[t0:t1])
            #totalpcs, ap_mns, ap_sd2s, _ap_Ns = _aoc2.GKs(oo.dat[t0:t1], oo.occ_bin)
            print("sum of ap_Ns  %.1f" % _N.sum(_ap_Ns))

            ap_isd2s = 1./ap_sd2s
            #totalpcs, _ap_Ns, ap_mns, ap_sd2s, ap_isd2s = _aoc2.approx_maze(oo.dat[t0:t1], oo.segs)


            if oo.fkpth_prms is not None:
                print("----------  oo.fkpth_prms is not None")
                ap_mn_x   = _N.array(ap_mns[:, 0].tolist() + oo.fkpth_prms[:, 0].tolist())
                ap_mn_y   = _N.array(ap_mns[:, 1].tolist() + oo.fkpth_prms[:, 1].tolist())
                ap_sd2s_x = _N.array(ap_sd2s[:, 0].tolist() + (oo.fkpth_prms[:, 2]**2).tolist())
                ap_sd2s_y = _N.array(ap_sd2s[:, 1].tolist() + (oo.fkpth_prms[:, 3]**2).tolist())
                
                fkL = oo.fkpth_wgt*oo.fkpth_prms[:, 4]
                #fkL = _N.ones(oo.fkpth_prms.shape[0])*wgts

                ap_Ns     = _N.array(_ap_Ns.tolist() + fkL.tolist())
                totalpcs += oo.fkpth_prms.shape[0]
                print("totalpcs  %d" % totalpcs)
            else:
                print("----------  oo.fkpth_prms not used")
                ap_mn_x   = _N.array(ap_mns[:, 0])
                ap_mn_y   = _N.array(ap_mns[:, 1])
                ap_sd2s_x = _N.array(ap_sd2s[:, 0])
                ap_sd2s_y = _N.array(ap_sd2s[:, 1])
                ap_Ns     = _N.array(_ap_Ns)

            # # DEBUG:  look at occupation and constraints#######

            # xlo = _N.min(oo.dat[t0:t1, 0])
            # xhi = _N.max(oo.dat[t0:t1, 0])
            # ylo = _N.min(oo.dat[t0:t1, 1])
            # yhi = _N.max(oo.dat[t0:t1, 1])
            # xA  = xhi-xlo
            # yA  = yhi-ylo
            # #BNS = 150
            # BNS = 500

            # #xms = _N.linspace(xlo - 0.05*xA, xhi + 0.05*xA, BNS+1, endpoint=True)
            # #yms = _N.linspace(ylo - 0.05*yA, yhi + 0.05*yA, BNS+1, endpoint=True)
            # xms = _N.linspace(xlo - 1.5*xA, xhi + 1.5*xA, BNS+1, endpoint=True)
            # yms = _N.linspace(ylo - 1.5*yA, yhi + 1.5*yA, BNS+1, endpoint=True)

            # xms_r  = xms.reshape((BNS+1, 1, 1))
            # yms_r  = yms.reshape((1, BNS+1, 1))
            # mn_x_r   = ap_mn_x.T.reshape((1, 1, totalpcs))
            # mn_y_r   = ap_mn_y.T.reshape((1, 1, totalpcs))            
            # sd2s_x_r  = ap_sd2s_x.T.reshape((1, 1, totalpcs))
            # sd2s_y_r  = ap_sd2s_y.T.reshape((1, 1, totalpcs))
            # ap_Ns_r  = ap_Ns.reshape((1, 1, totalpcs))

            # A      = ap_Ns_r / _N.sqrt(4*_N.pi*_N.pi*sd2s_x_r*sd2s_y_r)# * dx * dy

            # occ_xy = _N.sum(A*_N.exp(-0.5*(xms_r - mn_x_r)*(xms_r - mn_x_r)/sd2s_x_r - 0.5*(yms_r - mn_y_r)*(yms_r - mn_y_r)/sd2s_y_r), axis=2)
            # fig   = _plt.figure()
            # _plt.imshow(occ_xy.T, origin="lower", cmap=_plt.get_cmap("Blues"))

            # dx = _N.diff(xms)[0]
            # dy = _N.diff(yms)[0]

            # print("total  area   %.3f" % (_N.sum(occ_xy)*dx*dy))



            
            Asts    = _N.where(oo.dat[t0:t1, 2] == 1)[0]   #  based at 0   spike during movements
            print("number of spikes:   %d" % len(Asts))
            if len(Asts) < 10:
                print("nothing to do here, not enough spikes")
                return   #  not enough data, quit

            print(oo.dat.shape)

            if epc == ep1:   ###  initialize
                print("len(Asts)  %d" % len(Asts))
                print("t0   %(0)d     t1  %(1)d" % {"0" : t0, "1" : t1})

                #mving = _N.where(oo.dat[t0:t1, 8] == 1)[0] + t0
                #print("len mving  %d" % len(mving))
                labS, flatlabels, M_use = gAMxMu.initClusters(oo, M_max, K, xy, init_mks, t0, t1, Asts, xLo=oo.xLo, xHi=oo.xHi, oneCluster=oo.oneCluster, spcdim=2)

                #m1stSignalClstr = 0 if oo.oneCluster else nHSclusters[0]

                #  hyperparams of posterior. Not needed for all params
                u_u_  = _N.empty((M_use, K))
                u_Sg_ = _N.empty((M_use, K, K))
                Sg_nu_ = _N.empty(M_use)
                Sg_PSI_ = _N.zeros((M_use, K, K))

                #Sg_     = _N.empty((M_max, K, K))   # sampled value
                #uptriinds = _N.triu_indices_from(Sg[0],1)

                #######   containers for GIBBS samples iterations
                smp_sp_prms = _N.zeros((5, M_use, ITERS))  
                smp_mk_prms = [_N.zeros((M_use, ITERS, K)), 
                               _N.zeros((M_use, ITERS, K, K))]
                #  need mark hyp params cuz I calculate prior hyp from sampled hyps, unlike where I fit a distribution to sampled parameters and find best hyps from there.  Is there a better way?
                smp_mk_hyps = [_N.zeros((M_use, ITERS, K)),   
                               _N.zeros((M_use, ITERS, K, K)),
                               _N.zeros((M_use, ITERS)), 
                               _N.zeros((M_use, ITERS, K, K))]
                oo.smp_sp_prms = smp_sp_prms
                oo.smp_mk_prms = smp_mk_prms
                oo.smp_mk_hyps = smp_mk_hyps

                #####  MODES  - find from the sampling
                oo.sp_prmPstMd = _N.zeros(5*M_use)   # mode params
                oo.mk_prmPstMd = [_N.zeros((M_use, K)),
                                  _N.zeros((M_use, K, K))]
                      # mode of params


                #  list of freeClstrs
                freeClstr = _N.empty(M_max, dtype=_N.bool)   #  Actual cluster
                freeClstr[:] = True#False

                
                l0_M, fx_M, fy_M, q2x_M, q2y_M, u_M, Sg_M = gAMxMu.declare_params(M_max, K, spcdim=2)   #  nzclstr not inited  # sized to include noise cluster if needed
                l0_exp_hist_M = _N.empty(M_max)

                _l0_a_M, _l0_B_M, _fx_u_M, _fy_u_M, _fx_q2_M, _fy_q2_M, _q2x_a_M, _q2y_a_M, _q2x_B_M, _q2y_B_M, _u_u_M, \
                    _u_Sg_M, _Sg_nu_M, _Sg_PSI_M = gAMxMu.declare_prior_hyp_params(M_max, K, xy, fit_mks, Asts, t0, priors, labS, None, spcdim=2)

                    #_u_Sg_M, _Sg_nu_M, _Sg_PSI_M = gAMxMu.declare_prior_hyp_params(M_max, nHSclusters, K, xy, fit_mks, Asts, t0, priors, labS, labH, spcdim=2)

                print("----------------")
                print("M_use  %d" % M_use)
                
                l0, fx, fy, q2x, q2y, u, Sg        = gAMxMu.copy_slice_params_2d(M_use, l0_M, fx_M, fy_M, q2x_M, q2y_M, u_M, Sg_M)
                _l0_a, _l0_B, _fx_u, _fy_u, _fx_q2, _fy_q2, _q2x_a, _q2y_a, _q2x_B, _q2y_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI        = gAMxMu.copy_slice_hyp_params_2d(M_use, _l0_a_M, _l0_B_M, _fx_u_M, _fy_u_M, _fx_q2_M, _fy_q2_M, _q2x_a_M, _q2y_a_M, _q2x_B_M, _q2y_B_M, _u_u_M, _u_Sg_M, _Sg_nu_M, _Sg_PSI_M)

                print("l0------------")
                print(l0)


                print("----------------")
                print("M_use  %d" % M_use)
                
                l0_exp_hist = _N.array(l0_exp_hist_M[0:M_use], copy=True)

                gAMxMu.init_params_hyps_2d(oo, M_use, K, l0, fx, fy, q2x, q2y, u, Sg, _l0_a, _l0_B, _fx_u, _fy_u, _fx_q2, _fy_q2, _q2x_a, _q2y_a, _q2x_B, _q2y_B, _u_u, _u_Sg, _Sg_nu, \
                                        _Sg_PSI, Asts, t0, xy, fit_mks, flatlabels)#, nHSclusters)

                print("----------------")
                print("M_use  %d" % M_use)                
                
                ##   hard code
                #_q2x_u[0:(3*M)/4] = 

                U   = _N.empty(M_use)

                l0_exp_px_apprx = _N.empty(M_use)

                ######  the hyperparameters for f, q2, u, Sg, l0 during Gibbs
                #  f_u_, f_q2_, q2_a_, q2_B_, u_u_, u_Sg_, Sg_nu, Sg_PSI_, l0_a_, l0_B_
            else:
                #  later epochs

                freeInds = _N.where(freeClstr[0:M_use] == True)[0]
                n_fClstrs = len(freeInds)

                print("!!!!!!  %d" % n_fClstrs)
                print("bef M_use %d" % M_use)
                #  
                if n_fClstrs < oo.NExtrClstr:  #  
                    old_M = M_use
                    M_use  = M_use + (oo.NExtrClstr - n_fClstrs)
                    M_use = M_use if M_use < M_max else M_max
                    #new_M = M_use
                elif n_fClstrs > oo.NExtrClstr:
                    old_M = M_use
                    M_use  = M_use + (oo.NExtrClstr - n_fClstrs)
                    #new_M = M_use

                print("aft M_use %d" % M_use)

                l0, fx, fy, q2x, q2y, u, Sg        = gAMxMu.copy_slice_params_2d(M_use, l0_M, fx_M, fy_M, q2x_M, q2y_M, u_M, Sg_M)
                _l0_a, _l0_B, _fx_u, _fy_u, _fx_q2, _fy_q2, _q2x_a, _q2y_a, _q2x_B, _q2y_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI        = gAMxMu.copy_slice_hyp_params_2d(M_use, _l0_a_M, _l0_B_M, _fx_u_M, _fy_u_M, _fx_q2_M, _fy_q2_M, _q2x_a_M, _q2y_a_M, _q2x_B_M, _q2y_B_M, _u_u_M, _u_Sg_M, _Sg_nu_M, _Sg_PSI_M)                

                l0_exp_hist = _N.array(l0_exp_hist_M[0:M_use], copy=True)
                l0_exp_px_apprx = _N.empty(M_use)

                #  hyperparams of posterior. Not needed for all params
                u_u_  = _N.empty((M_use, K))
                u_Sg_ = _N.empty((M_use, K, K))
                Sg_nu_ = _N.empty(M_use)
                Sg_PSI_ = _N.empty((M_use, K, K))

                smp_sp_prms = _N.zeros((5, M_use, ITERS))  
                smp_mk_prms = [_N.zeros((M_use, ITERS, K)), 
                               _N.zeros((M_use, ITERS, K, K))]
                #  need mark hyp params cuz I calculate prior hyp from sampled hyps, unlike where I fit a distribution to sampled parameters and find best hyps from there.  Is there a better way?
                smp_mk_hyps = [_N.zeros((M_use, ITERS, K)),   
                               _N.zeros((M_use, ITERS, K, K)),
                               _N.zeros((M_use, ITERS)), 
                               _N.zeros((M_use, ITERS, K, K))]


                oo.smp_sp_prms = smp_sp_prms
                oo.smp_mk_prms = smp_mk_prms
                oo.smp_mk_hyps = smp_mk_hyps

                #####  MODES  - find from the sampling
                oo.sp_prmPstMd = _N.zeros(5*M_use)   # mode params
                oo.mk_prmPstMd = [_N.zeros((M_use, K)),
                                  _N.zeros((M_use, K, K))]

            NSexp   = t1-t0    #  length of position data  #  # of no spike positions to sum

            ######  we need these defined for position of spikes
            xt0t1 = _N.array(xy[t0:t1, 0])#smthd_pos
            yt0t1 = _N.array(xy[t0:t1, 1])#smthd_pos

            l0_exp_px = _N.empty(M_use)

            nSpks    = len(Asts)
            v_sts = _N.empty(len(Asts), dtype=_N.int)            
            #v_sts = _N.array(Asts)
            clstszs = _N.zeros((ITERS, M_use), dtype=_N.int)
            clstszs_rr  = clstszs.reshape((ITERS, M_use, 1, 1))
            clstszs_r  = clstszs.reshape((ITERS, M_use, 1))
            

            cls_str_ind = _N.zeros(M_use+1, dtype=_N.int)
            #cls_len      = _N.zeros(M_use, dtype=_N.int)

            if M_use > 1:
                cgz   = _N.zeros((ITERS, nSpks), dtype=_N.uint8)
                #gz   = _N.zeros((ITERS, nSpks, M_use), dtype=_N.uint8)
            else:
                cgz   = _N.ones((ITERS, nSpks), dtype=_N.uint8)
                            
                #gz   = _N.ones((ITERS, nSpks, M_use), dtype=_N.uint8)
                cls_str_ind[0] = 0
                cls_str_ind[1] = nSpks
                #cls_len[0] = nSpks
                v_sts = Asts + t0
                clstszs[:, 0] = nSpks
            oo.cgz=cgz

            xAS  = _N.array(xy[Asts + t0, 0])   #  position @ spikes.  creates new copy
            yAS  = _N.array(xy[Asts + t0, 1])   #  position @ spikes.  creates new copy
            mAS  = fit_mks[Asts + t0]   #  position @ spikes

            #fig = _plt.figure()
            
            #_plt.hist2d(xAS, yAS, bins=[xms, yms], cmap=_plt.get_cmap("Blues"))
            #fig = _plt.figure()
            #_plt.hist2d(xt0t1, yt0t1, bins=[xms, yms], cmap=_plt.get_cmap("Blues"))

            econt = _N.empty((M_use, nSpks))
            rat   = _N.zeros((M_use+1, nSpks))

            qdrMKS = _N.empty((M_use, nSpks))
            qdrSPC = _N.empty((M_use, nSpks))            
            exp_arg= _N.empty((M_use, nSpks))
            ################################  GIBBS ITERS ITERS ITERS

            _iu_Sg = _N.array(_u_Sg)
            for m in range(M_use):
                _iu_Sg[m] = _N.linalg.inv(_u_Sg[m])

            ttA = _tm.time()

            _Dl0_a = _N.empty(M_use);            _Dl0_B = _N.empty(M_use)


            mcs = _N.empty((M_use, K))   # cluster sample means
            mcsT = _N.empty((M_use, K))   # cluster sample means
            outs1 = _N.empty((M_use, K))
            outs2 = _N.empty((M_use, K))

            BLK        = 1000
            iterBLOCKs = ITERS//BLK   # new in python 3  //  int/int = int

            q2x = _N.array([0.02]*M_use)
            q2y = _N.array([0.02]*M_use)

            inv_sum_sd2s_x = _N.empty((M_use, totalpcs))
            inv_sum_sd2s_y = _N.empty((M_use, totalpcs))

            nrm_x          = _N.empty((M_use, totalpcs))
            nrm_y          = _N.empty((M_use, totalpcs))
            diff2_x        = _N.empty((M_use, totalpcs))
            diff2_y        = _N.empty((M_use, totalpcs))
            

            _cdfs2dA.nrm_xy(totalpcs, M_use, inv_sum_sd2s_x, nrm_x, q2x, ap_sd2s_x)
            _cdfs2dA.nrm_xy(totalpcs, M_use, inv_sum_sd2s_y, nrm_y, q2y, ap_sd2s_y)
            _cdfs2dA.diff2_xy(totalpcs, M_use, diff2_x, fx, ap_mn_x)
            _cdfs2dA.diff2_xy(totalpcs, M_use, diff2_y, fy, ap_mn_y)

            # print(_N.sum(_N.isnan(nrm_x)))
            # print(_N.sum(_N.isinf(nrm_x)))            
            # print(_N.sum(_N.isnan(nrm_y)))
            # print(_N.sum(_N.isinf(nrm_y)))
            # print(_N.sum(_N.isnan(inv_sum_sd2s_x)))
            # print(_N.sum(_N.isinf(inv_sum_sd2s_x)))            
            # print(_N.sum(_N.isnan(inv_sum_sd2s_y)))
            # print(_N.sum(_N.isinf(inv_sum_sd2s_y)))            
            
            goback = 500

            ###########  BEGIN GIBBS SAMPLING ##############################

            max_so_far = _N.empty(nSpks)

            # l0[:] = 20000
            # q2x[:] = 16
            # q2y[:] = 9
            for itrB in range(iterBLOCKs):
                for itr in range(itrB*BLK, (itrB+1)*BLK):
                    #ttsw1 = _tm.time()
                    iSg = _N.linalg.inv(Sg)
                    #ttsw2 = _tm.time()
                    if (itr % 100) == 0:    
                        print("-------itr  %(i)d" % {"i" : itr})

                    if M_use > 1:
                        _sA.stochasticAssignment_2d(epc, itr, M_use, K, l0, fx, fy, q2x, q2y, u, Sg, iSg, nSpks, t0, mAS, xAS, yAS, rat, econt, max_so_far, cgz, qdrSPC, qdrMKS, exp_arg, freeClstr, clstszs[itr])
                        _fm.cluster_bounds2_cgz(clstszs[itr], Asts, cls_str_ind, v_sts, cgz[itr], t0, M_use, nSpks)    # _fm.cluser_bounds provides no improvement
                    #ttsw3 = _tm.time()
                    #ttsw4 = _tm.time()

                    ###############
                    ###############     u
                    ###############
                    _N.copyto(u_Sg_, _N.linalg.inv(_iu_Sg + clstszs_rr[itr]*iSg))
                    _fm.find_mcs(clstszs[itr], v_sts, cls_str_ind, fit_mks, mcs, M_use, K)
                    _fm.multiple_mat_dot_v(_iu_Sg, _u_u, outs1, M_use, K)
                    _fm.multiple_mat_dot_v(iSg, mcs, outs2, M_use, K)
                    _fm.multiple_mat_dot_v(u_Sg_, outs1 + clstszs_r[itr]*outs2, u_u_, M_use, K)

                    #ttsw5 = _tm.time()
                    ucmvnrms= _N.random.randn(M_use, K)

                    try:
                        C       = _N.linalg.cholesky(u_Sg_)
                    except _N.linalg.linalg.LinAlgError:
                        dmp = open("cholesky.dmp", "wb")
                        pickle.dump([u_Sg_, _iu_Sg, clstszs[itr], iSg, _u_Sg, _u_u], dmp, -1)
                        dmp.close()

                        raise
                    u       = _N.einsum("njk,nk->nj", C, ucmvnrms) + u_u_
                    # print("-----  u")
                    # print(u)

                    smp_mk_prms[oo.ky_p_u][:, itr] = u  # dim of u wrong
                    smp_mk_hyps[oo.ky_h_u_u][:, itr] = u_u_
                    smp_mk_hyps[oo.ky_h_u_Sg][:, itr] = u_Sg_


                    #ttsw6 = _tm.time()
                    ###############
                    ###############  Conditional f
                    ###############
                    if (epc > 0) and oo.adapt:
                        q2xpr = _fx_q2 + f_q2_rate * DT
                        q2ypr = _fy_q2 + f_q2_rate * DT
                    else:
                        q2xpr = _fx_q2
                        q2ypr = _fy_q2

                    m_rnds_x = _N.random.rand(M_use)
                    m_rnds_y = _N.random.rand(M_use)

                    #print("b4 fx %.2f" % fx[0])
                    _cdfs2dA.smp_f(0, itr, M_use, xt0t1, clstszs[itr], cls_str_ind, 
                                   v_sts, t0, l0,
                                   totalpcs, ap_Ns, ap_mn_x,
                                   diff2_y, nrm_y, inv_sum_sd2s_y,
                                   diff2_x, nrm_x, inv_sum_sd2s_x,
                                   _fx_u, _fx_q2, m_rnds_x, fx, q2x, 0)
                    smp_sp_prms[oo.ky_p_fx, :, itr]   = fx

                    _cdfs2dA.smp_f(1, itr, M_use, yt0t1, clstszs[itr], cls_str_ind, 
                                        v_sts, t0, l0,
#                                   totalpcs, ap_mn_y, 
                                        totalpcs, ap_Ns, ap_mn_y, 
                                        diff2_x, nrm_x, inv_sum_sd2s_x,
                                        diff2_y, nrm_y, inv_sum_sd2s_y,
                                        _fy_u, _fy_q2, m_rnds_y, fy, q2y, 0)

                    smp_sp_prms[oo.ky_p_fy, :, itr]   = fy
                    
                    #ttsw7 = _tm.time()
                    ##############
                    ##############  VARIANCE, COVARIANCE
                    ##############

                    # tt6a = 0
                    # tt6b = 0
                    # tt6c = 0

                    Sg_nu_ = _Sg_nu + clstszs[itr]
                    _fm.Sg_PSI(cls_str_ind, clstszs[itr], v_sts, fit_mks, _Sg_PSI, Sg_PSI_, u, M_use, K)                                        
                    Sg = _iw.invwishartrand_multi_tempM(M_use, K, Sg_nu_, Sg_PSI_)

                    smp_mk_prms[oo.ky_p_Sg][:, itr] = Sg
                    smp_mk_hyps[oo.ky_h_Sg_nu][:, itr] = Sg_nu_
                    smp_mk_hyps[oo.ky_h_Sg_PSI][:, itr] = Sg_PSI_

                    #ttsw8 = _tm.time()
                    ##############
                    ##############  SAMPLE SPATIAL VARIANCE
                    ##############
                    #  B' / (a' - 1) = MODE   #keep mode the same after discount


                    mx_rnds = _N.random.rand(M_use) 
                    my_rnds = _N.random.rand(M_use)
                    #  B' = MODE * (a' - 1)
                    if (epc > 0) and oo.adapt:
                        _mdx_nd= _q2x_B / (_q2x_a + 1)
                        _Dq2x_a = _q2x_a * _N.exp(-DT/tau_q2)
                        _Dq2x_B = _mdx_nd * (_Dq2x_a + 1)
                        _mdy_nd= _q2y_B / (_q2y_a + 1)
                        _Dq2y_a = _q2y_a * _N.exp(-DT/tau_q2)
                        _Dq2y_B = _mdy_nd * (_Dq2y_a + 1)
                    else:
                        _Dq2x_a = _q2x_a
                        _Dq2x_B = _q2x_B
                        _Dq2y_a = _q2y_a
                        _Dq2y_B = _q2y_B

                    #ttsw9 = _tm.time()

                    m_rnds_x = _N.random.rand(M_use)
                    m_rnds_y = _N.random.rand(M_use)

                    #  B' = MODE * (a' - 1)

                    _cdfs2dA.smp_q2(0, itr, M_use, xt0t1, clstszs[itr], cls_str_ind, 
                                    v_sts, t0, l0,
#                                    totalpcs, ap_sd2s_x,
                                    totalpcs, ap_Ns, ap_sd2s_x,
                                    diff2_y, nrm_y, inv_sum_sd2s_y,
                                    diff2_x, nrm_x, inv_sum_sd2s_x,
                                    _Dq2x_a, _Dq2x_B, m_rnds_x, fx, q2x, 0)

                    smp_sp_prms[oo.ky_p_q2x, :, itr]   = q2x

                    _cdfs2dA.smp_q2(1, itr, M_use, yt0t1, clstszs[itr], cls_str_ind, 
                                    v_sts, t0, l0,
#                                    totalpcs, ap_sd2s_y,
                                    totalpcs, ap_Ns, ap_sd2s_y,
                                    diff2_x, nrm_x, inv_sum_sd2s_x,
                                    diff2_y, nrm_y, inv_sum_sd2s_y,
                                    _Dq2y_a, _Dq2y_B, m_rnds_y, fy, q2y, 0)
                    smp_sp_prms[oo.ky_p_q2y, :, itr]   = q2y
#                     #ttsw10 = _tm.time()

                    ###############
                    ###############  CONDITIONAL l0
                    ###############
                    #  _ss.gamma.rvs.  uses k, theta  k is 1/B (B is our thing)

                    _cdfs2dA.l0_spatial(M_use, totalpcs, oo.dt, ap_Ns, nrm_x, nrm_y, diff2_x, diff2_y, inv_sum_sd2s_x, inv_sum_sd2s_y, l0_exp_px_apprx)
                    #print("**********   l0_exp_px_apprx")


                    
                    BL  = l0_exp_px_apprx    #  dim M

                    if (epc > 0) and oo.adapt:
                        _mn_nd= _l0_a / _l0_B
                        #  variance is a/b^2
                        #  a/2 / B/2    variance is a/2 / B^2/4 = 2a^2 / B^2  
                        #  variance increases by 2

                        _Dl0_a = _l0_a * _N.exp(-DT/tau_l0)
                        _Dl0_B = _Dl0_a / _mn_nd
                    else:
                        _Dl0_a = _l0_a
                        _Dl0_B = _l0_B

                    aL  = clstszs[itr] + 1     #  problem:  if clstsz = 0
                    l0_a_ = aL + _Dl0_a
                    l0_B_ = BL + _Dl0_B

                    l0 = _ss.gamma.rvs(l0_a_, scale=(1/l0_B_), size=M_use)  #  check                    

                    smp_sp_prms[oo.ky_p_l0, :, itr] = l0

                    #ttsw11 = _tm.time()
                    # print "#timing start"
                    # print "nt+= 1"
                    # print "t2t1+=%.4e" % (#ttsw2-#ttsw1)
                    # print "t3t2+=%.4e" % (#ttsw3-#ttsw2)
                    # print "t4t3+=%.4e" % (#ttsw4-#ttsw3)
                    # print "t5t4+=%.4e" % (#ttsw5-#ttsw4)
                    # print "t6t5+=%.4e" % (#ttsw6-#ttsw5)
                    # print "t7t6+=%.4e" % (#ttsw7-#ttsw6)  # slow
                    # print "t8t7+=%.4e" % (#ttsw8-#ttsw7)
                    # print "t9t8+=%.4e" % (#ttsw9-#ttsw8)
                    # print "t10t9+=%.4e" % (#ttsw10-#ttsw9)
                    # print "t11t10+=%.4e" % (#ttsw11-#ttsw10)
                    # print "#timing end  %.5f" % (#ttsw10-#ttsw1)


                tFinishBLK = _tm.time()
                #print("FINISH BLCK %(tet)d   Last 1000 iters:  %(tm).2f" % {"tet" : oo.tetr, "tm" : (tFinishBLK - tEnterBLK)})                
                stop_Gibbs = False
                cond1 = ((epc == 0) and (itr >= 6000))
                cond2 = ((epc > 0) and (itr >= 4000))


                if (global_stop_condition and (cond1 or cond2)):
                    print("I SHOULD NEVER BE HERE")
                    tttt1 = _tm.time()
                    stop_Gibbs = _pU.stop_Gibbs_cgz(itr, M_use, nSpks, smp_sp_prms, clstszs, spcdim=2)
                    tttt2 = _tm.time()
                    print("tet %(tet)d   stop_Gibbs:  %(tm).2f" % {"tet" : oo.tetr, "tm" : (tttt2-tttt1)})
                    
                if stop_Gibbs:
                    print(global_stop_condition)
                    print("!!!!!  tetr %(tet)d   stop Gibbs at %(itr)d" % {"tet" : oo.tetr, "itr" : itr})
                    goback = 3000
                    break
                    

                #frms = _pU.find_good_clstrs_and_stationary_from(M_use, smp_sp_prms[:, 0:itr+1])
                #if (itr >= oo.earliest) and (len(_N.where(frms - 4000 < 0)[0]) == M_use):
                #    break

            ttB = _tm.time()
            print (ttB-ttA)

            print("done -------- ")
            print(_N.mean(diff2_x))
            print(_N.std(diff2_x))
            print(_N.mean(diff2_y))
            print(_N.std(diff2_y))
            print("** -------- ")
            print(_N.mean(inv_sum_sd2s_x))
            print(_N.mean(inv_sum_sd2s_y))
            print(_N.std(inv_sum_sd2s_x))
            print(_N.std(inv_sum_sd2s_y))
            print(_N.mean(nrm_x))
            print(_N.mean(nrm_y))            
            print(_N.std(nrm_x))
            print(_N.std(nrm_y))            


            print(u)
            print(Sg)
            print("=========================")
            print(l0)
            print(fx)
            print(fy)
            print(q2x)
            print(q2y)            
            
            print("itr is %d" % itr)

            gAMxMu.finish_epoch2_cgz_2d(oo, nSpks, epc, itr+1, clstszs, l0, fx, fy, q2x, q2y, u, Sg, _fx_u, _fy_u, _fx_q2, _fy_q2, _q2x_a, _q2y_a, _q2x_B, _q2y_B, _l0_a, _l0_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, smp_sp_prms, smp_mk_prms, smp_mk_hyps, freeClstr, M_use, K, priors, global_stop_condition, goback//3)
            
            #  _l0_a is a copy of a subset of _l0_a_M
            #  we need to copy back the values _l0_a back into _l0_a_M
            gAMxMu.contiguous_inuse_cgz_2d(M_use, M_max, K, freeClstr, l0, fx, fy, q2x, q2y, u, Sg, _l0_a, _l0_B, _fx_u, _fy_u, _fx_q2, _fy_q2, _q2x_a, _q2y_a, _q2x_B, _q2y_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, smp_sp_prms, smp_mk_prms, oo.sp_prmPstMd, oo.mk_prmPstMd, cgz, priors)

            gAMxMu.copy_back_params_2d(M_use, l0, fx, fy, q2x, q2y, u, Sg, M_max, l0_M, fx_M, fy_M, q2x_M, q2y_M, u_M, Sg_M)
            gAMxMu.copy_back_hyp_params_2d(M_use, _l0_a, _l0_B, _fx_u, _fy_u, _fx_q2, _fy_q2, _q2x_a, _q2y_a, _q2x_B, _q2y_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, M_max, _l0_a_M, _l0_B_M, _fx_u_M, _fy_u_M, _fx_q2_M, _fy_q2_M, _q2x_a_M, _q2y_a_M, _q2x_B_M, _q2y_B_M, _u_u_M, _u_Sg_M, _Sg_nu_M, _Sg_PSI_M)
            
            #  MAP of nzclstr
            if saveSamps:
                pcklme["smp_sp_prms"] = smp_sp_prms[:, 0:itr+1]
                pcklme["smp_mk_prms"] = [smp_mk_prms[0][:, 0:itr+1], smp_mk_prms[1][:, :, 0:itr+1]]
            pcklme["sp_prmPstMd"] = oo.sp_prmPstMd
            pcklme["mk_prmPstMd"] = oo.mk_prmPstMd
            pcklme["intvs"]       = oo.intvs
            if saveOcc:
                pcklme["occ"]         = cgz[0:itr+1]                
                pcklme["freeClstr"]           = freeClstr  #  next time
            pcklme["nz_pth"]         = nz_pth
            pcklme["M"]           = M_use
            # pcklme["t0_0"]        = oo.t0_0
            # pcklme["t1_0"]        = oo.t1_0
            # pcklme["t1_1"]        = oo.t1_1
            # pcklme["t0_1"]        = oo.t0_1

            dmp = open(resFN("posteriors_%d.dmp" % epc, dir=oo.outdir, __JIFIResultDir__=oo.__JIFIResultDir__), "wb")
            pickle.dump(pcklme, dmp, -1)
            dmp.close()

        print("DONE WITH GIBBS GIBBS 2d  %(tet)d   %(save)s" % {"tet" : oo.tetr, "save" : oo.outdir})
