"""
V1.2   use adaptive range for integrating over f
variance   0.0001 
"""
import stats_util as s_u
import scipy.stats as _ss
import os
import time as _tm
import numpy as _N
import matplotlib.pyplot as _plt
from   sumojam.tools.SUMOdirs import resFN, datFN
import pickle
import sumojam.posteriorUtil as _pU
import sumojam.gibbs1d_util as gAMxMu
import sumojam.stochasticAssignment as _sA
import sumojam.cdf_smp_1d as _cdfs
import sumojam.fastnum as _fm
import sumojam.tools.compress_gz as c_gz
import sumojam.iwish as _iw
import sumojam.tools.anocc as _aoc2


class GibbsSampler1D:
    ky_p_l0 = 0;    ky_p_f  = 1;    ky_p_q2 = 2
    ky_h_l0_a = 0;  ky_h_l0_B=1;
    ky_h_f_u  = 2;  ky_h_f_q2=3;
    ky_h_q2_a = 4;  ky_h_q2_B=5;

    ky_p_u = 0;       ky_p_Sg = 1;
    ky_h_u_u = 0;     ky_h_u_Sg=1;
    ky_h_Sg_nu = 2;   ky_h_Sg_PSI=3;

    dt      = 0.001
    #  position dependent firing rate
    ######################################  PRIORS
    twpi = 2*_N.pi
    #NExtrClstr = 5
    NExtrClstr = 3
    earliest   = 20000      #   min # of gibbs samples

    #  sizes of arrays
    NposHistBins = 200      #   # points to sample position with  (uniform lam(x)p(x))

    fkpth_prms = None
    intvs = None    #  
    dat   = None

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

    #  px and spatial integration limits.  Don't make larger accessible space
    xLo      = -6;    xHi      = 6   

    #  limits of conditional probability sampling of f and q2
    f_L   = -12;     f_H = 12   
    q2_L = 1e-6;    q2_H = 1e4

    oneCluster = False
    tetr       = None
    __JIFIResultDir__ = None

    priors     = None

    def setup_spatial_sum_params(self, q2x=None, fx=None): 
        oo = self
        oo.q2_L, oo.q2_H = q2x   #  grid lims for calc of conditional posterior
        oo.f_L,  oo.f_H  = fx
        

    def __init__(self, outdir, fn, intvfn, xLo, xHi, adapt, oneCluster=False, tet=None, __JIFIResultDir__=None, seg_filtered_intvs=None, spc_2d_use_1d_col=-1, seg_col=None, mv_stop_col=None):
        print("gibbs init    %s" % fn)
        oo     = self
        oo.__JIFIResultDir__=__JIFIResultDir__
        oo.t0_0 = _tm.time()
        
        oo.tetr = -1 if tet is None else tet
        oo.oneCluster = oneCluster
        oo.adapt = adapt
        #_N.random.seed(seed)

        ######################################  DATA input, define intervals
        # bFN = fn[0:-4]
        oo.outdir = outdir

        _dat    = _N.loadtxt("%s.dat" % datFN(fn, create=False))

        if spc_2d_use_1d_col != -1:  #  x y s01 m1 m2 m3 m4 (seg) (stop_mv)
            if mv_stop_col is not None:   #  data has move and stop
                mvng_col = 2+1+4 + (seg_col is not None) + 1- 1
                mving_ts = _N.where(_dat[:, mvng_col] == 1)[0]
            else:
                mving_ts = _N.arange(_dat.shape[0])
            oo.dat  = _N.empty((len(mving_ts), _dat.shape[1] - 1))
            oo.dat[:, 0]  = _dat[mving_ts, spc_2d_use_1d_col]
            oo.dat[:, 1:]  = _dat[mving_ts, 2:]
        else:  #  x s01 m1 m2 m3 m4 (seg) (stop_mv)
            if mv_stop_col is not None:   #  data has move and stop
                mvng_col = 1+1+4 + (seg_col is not None) + 1 - 1
                mving_ts = _N.where(_dat[:, mvng_col] == 1)[0]
            oo.dat  = _N.array(_dat[mving_ts])
            
        #oo.dat    = _N.loadtxt("%s.dat" % datFN(fn, create=False))

        intvs     = _N.loadtxt("%s.dat" % datFN(intvfn, create=False))
        _intvs  = _N.array(intvs*_dat.shape[0], dtype=_N.int) if seg_filtered_intvs is None else seg_filtered_intvs

        oo.intvs  = _N.array(_intvs)  # intervals in time of segment filtered data

        t0 = 0
        for i in range(len(intvs)-1):

            t1 = t0 + _N.sum(_dat[_intvs[i]:_intvs[i+1], mvng_col])   #  only moving 
            print("*****  %(0)d  %(1)d" % {"0" : t0, "1" : t1})
            oo.intvs[i] = t0
            oo.intvs[i+1] = t1
            t0 = t1
        oo.epochs    = oo.intvs.shape[0] - 1
        
        NT     = oo.dat.shape[0]
        oo.xLo = xLo
        oo.xHi = xHi


    def gibbs(self, ITERS, K, priors, ep1, ep2, saveSamps, saveOcc, nz_pth=0., smth_pth_ker=0, f_STEPS=13, q2_STEPS=13, f_SMALL=10, q2_SMALL=10, f_cldz=10, q2_cldz=10, minSmps=20, occ_bins=30, fkpth_wgt=0, global_stop_condition=False, ignore_space=0):
        """
        gtdiffusion:  use ground truth center of place field in calculating variance of center.  Meaning of diffPerMin different
        """
        oo = self
        print("RUNNING GIBBS GIBBS 1d  %(tet)d   %(save)s" % {"tet" : oo.tetr, "save" : oo.outdir})                

        twpi     = 2*_N.pi
        pcklme   = {}

        oo.priors = priors

        ep2 = oo.epochs if (ep2 == None) else ep2
        oo.epochs = ep2-ep1

        freeClstr = None

        x      = oo.dat[:, 0]
        xr    = x.reshape((x.shape[0], 1))

        init_mks    = _N.array(oo.dat[:, 2:2+K])  #  init using non-rot
        fit_mks    = init_mks

        f_q2_rate = (oo.diffusePerMin**2)/60000.  #  unit of minutes  
        
        ######################################  PRECOMPUTED

        #_ifcp.init()
        tau_l0 = oo.t_hlf_l0/_N.log(2)
        tau_q2 = oo.t_hlf_q2/_N.log(2)

        _cdfs.init(oo.dt, oo.f_L, oo.f_H, oo.q2_L, oo.q2_H, f_STEPS, q2_STEPS, f_SMALL, q2_SMALL, f_cldz, q2_cldz, minSmps)

        M_max   = 50   #  100 clusters, max
        M_use    = 0     #  number of non-free + 5 free clusters

        for epc in range(ep1, ep2):
            print("^^^^^^^^^^^^^^^^^^^^^^^^    tet %(t)d  epoch %(e)d" % {"t" : oo.tetr, "e" : epc})

            t0 = oo.intvs[epc]
            t1 = oo.intvs[epc+1]
            print("t0   %(0)d     t1  %(1)d" % {"0" : t0, "1" : t1})
                
            
            #oo.dat[t0:t1, 0] += nz_pth*_N.random.randn(len(oo.dat[t0:t1, 0]))

            if epc > 0:
                tm1= oo.intvs[epc-1]
                #  0 10 30     20 - 5  = 15    0.5*((10+30) - (10+0)) = 15
                DT = t0-tm1

            posbins  = _N.linspace(oo.xLo, oo.xHi, oo.Nupx+1)
            
            xt0t1 = x[t0:t1]#smthd_pos

            Asts    = _N.where(oo.dat[t0:t1, 1] == 1)[0]   #  based at 0
            nSpks    = len(Asts)
            print("!!!!  nSpks %d" % nSpks)

            totalpcs, ap_mns, ap_sd2s, _ap_Ns = _aoc2.approx_maze_1D_smthd(oo.dat[t0:t1], BNS=occ_bins, smth_krnl=2, nz_pth=nz_pth)
            
            #totalpcs, ap_mns, ap_sd2s, _ap_Ns = _aoc2.approx_maze_1D_smthd(oo.dat[t0:t1], BNS=occ_bins, smth_krnl=2)
            #totalpcs, ap_mns, ap_sd2s, _ap_Ns = _aoc2.approx_maze_1D(oo.dat[t0:t1], BNS=occ_bins)

            if fkpth_wgt > 0:
                print("----------  oo.fkpth_prms is not None")
                ap_mn_x   = _N.array(ap_mns[0].tolist() + [-12, 12])
                ap_sd2s_x = _N.array(ap_sd2s[:, 0].tolist() + [0.3, 0.3])

                fkL = fkpth_wgt*(t1-t0)

                ap_Ns     = _N.array(_ap_Ns.tolist() + [fkL, fkL])
                totalpcs += 2
                print("totalpcs  %d" % totalpcs)
            else:
                print("----------  oo.fkpth_prms not used")

                ap_mn_x   = ap_mns
                ap_sd2s_x = ap_sd2s
                ap_Ns     = _ap_Ns
                    
            
            if epc == ep1:   ###  initialize
                labS, flatlabels, M_use = gAMxMu.initClusters(oo, M_max, K, xr, init_mks, t0, t1, Asts, xLo=oo.xLo, xHi=oo.xHi, oneCluster=oo.oneCluster)
                print("flatlabels")
                print(flatlabels)

                #  hyperparams of posterior. Not needed for all params
                u_u_  = _N.empty((M_use, K))
                u_Sg_ = _N.empty((M_use, K, K))
                Sg_nu_ = _N.empty(M_use)
                Sg_PSI_ = _N.zeros((M_use, K, K))
                Sg_PSI_diag = _N.zeros((M_use, K))

                #######   containers for GIBBS samples iterations
                smp_sp_prms = _N.zeros((3, M_use, ITERS))  
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
                oo.sp_prmPstMd = _N.zeros(3*M_use)   # mode params
                oo.mk_prmPstMd = [_N.zeros((M_use, K)),
                                  _N.zeros((M_use, K, K))]
                      # mode of params

                #  list of freeClstrs
                freeClstr = _N.empty(M_max, dtype=_N.bool)   #  Actual cluster
                freeClstr[:] = True#False

                
                l0_M, f_M, q2_M, u_M, Sg_M = gAMxMu.declare_params(M_max, K)   #  nzclstr not inited  # sized to include noise cluster if needed
                l0_exp_hist_M = _N.empty(M_max)
                _l0_a_M, _l0_B_M, _f_u_M, _f_q2_M, _q2_a_M, _q2_B_M, _u_u_M, \
                    _u_Sg_M, _Sg_nu_M, _Sg_PSI_M = gAMxMu.declare_prior_hyp_params(M_max, K, x, fit_mks, Asts, t0, priors, labS, None)                    

                l0, f, q2, u, Sg        = gAMxMu.copy_slice_params(M_use, l0_M, f_M, q2_M, u_M, Sg_M)
                _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI        = gAMxMu.copy_slice_hyp_params(M_use, _l0_a_M, _l0_B_M, _f_u_M, _f_q2_M, _q2_a_M, _q2_B_M, _u_u_M, _u_Sg_M, _Sg_nu_M, _Sg_PSI_M)

                l0_exp_hist = _N.array(l0_exp_hist_M[0:M_use], copy=True)

                fr = f.reshape((M_use, 1))
                gAMxMu.init_params_hyps(oo, M_use, K, l0, f, q2, u, Sg, _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, \
                    _Sg_PSI, Asts, t0, x, fit_mks, flatlabels)

                U   = _N.empty(M_use)
                pr_u_Sg = _u_Sg  #  ad hoc for now
                i_pr_u_Sg = _N.array(pr_u_Sg)
                for m in range(M_use):
                    i_pr_u_Sg[m] = _N.linalg.inv(pr_u_Sg[m])
                
                l0_exp_px_apprx = _N.empty(M_use)
            else:
                #  later epochs

                freeInds = _N.where(freeClstr[0:M_use] == True)[0]
                n_fClstrs = len(freeInds)
                print(freeClstr)
                print("!!!!!!  %d" % n_fClstrs)
                print("bef M_use %d" % M_use)
                #

                if (not oo.oneCluster) and (n_fClstrs < oo.NExtrClstr):  #  
                    old_M = M_use
                    M_use  = M_use + (oo.NExtrClstr - n_fClstrs)
                    M_use = M_use if M_use < M_max else M_max
                    #new_M = M_use
                elif (not oo.oneCluster) and (n_fClstrs > oo.NExtrClstr):
                    old_M = M_use
                    M_use  = M_use + (oo.NExtrClstr - n_fClstrs)
                    #new_M = M_use

                print("aft M_use %d" % M_use)
                print(freeClstr)
                M_1st_free = _N.where(freeClstr[0:M_use] == False)[0][-1] + 1
                print("M_not_free %d" % M_1st_free)                

                l0, f, q2, u, Sg        = gAMxMu.copy_slice_params(M_use, l0_M, f_M, q2_M, u_M, Sg_M)
                _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI        = gAMxMu.copy_slice_hyp_params(M_use, _l0_a_M, _l0_B_M, _f_u_M, _f_q2_M, _q2_a_M, _q2_B_M, _u_u_M, _u_Sg_M, _Sg_nu_M, _Sg_PSI_M)

                l0_exp_hist = _N.array(l0_exp_hist_M[0:M_use], copy=True)
                l0_exp_px_apprx = _N.empty(M_use)

                #  hyperparams of posterior. Not needed for all params
                u_u_  = _N.empty((M_use, K))
                u_Sg_ = _N.empty((M_use, K, K))
                Sg_nu_ = _N.empty(M_use)
                Sg_PSI_ = _N.empty((M_use, K, K))
                Sg[M_use-oo.NExtrClstr:M_use] = _N.eye(K)   #  Force to be pos. def.                

                ###  add uncertainty to _u_Sg (Do this once per epoch)
                # print("!!!!!!!!!")
                # print(_u_Sg)
                # print(_u_u)
                # print(_f_u)
                # print(_f_q2)
                # print("_-_-_-_-_-_-_-")
                pr_u_Sg = _u_Sg#*2  #  ad hoc for now
                i_pr_u_Sg = _N.array(pr_u_Sg)
                for m in range(M_use):
                    i_pr_u_Sg[m] = _N.linalg.inv(pr_u_Sg[m])

                #######   containers for GIBBS samples iterations
                #smp_sp_prms = _N.zeros((3, ITERS, M_use))
                smp_sp_prms = _N.zeros((3, M_use, ITERS))  
                #smp_mk_prms = [_N.zeros((K, ITERS, M_use)), 
                #               _N.zeros((K, K, ITERS, M_use))]
                smp_mk_prms = [_N.zeros((M_use, ITERS, K)), 
                               _N.zeros((M_use, ITERS, K, K))]
                
                smp_mk_hyps = [_N.zeros((M_use, ITERS, K)),   
                               _N.zeros((M_use, ITERS, K, K)),
                               _N.zeros((M_use, ITERS)), 
                               _N.zeros((M_use, ITERS, K, K))]
                
                oo.smp_sp_prms = smp_sp_prms
                oo.smp_mk_prms = smp_mk_prms
                oo.smp_mk_hyps = smp_mk_hyps

                #####  MODES  - find from the sampling
                oo.sp_prmPstMd = _N.zeros(3*M_use)   # mode params
                oo.mk_prmPstMd = [_N.zeros((M_use, K)),
                                  _N.zeros((M_use, K, K))]

            NSexp   = t1-t0    #  length of position data  #  # of no spike positions to sum
            xt0t1 = _N.array(x[t0:t1])

            v_sts = _N.empty(nSpks, dtype=_N.int)
            cls_str_ind = _N.zeros(M_use+1, dtype=_N.int)
            clstszs = _N.zeros((ITERS, M_use), dtype=_N.int)
            clstszs_rr  = clstszs.reshape((ITERS, M_use, 1, 1))
            clstszs_r  = clstszs.reshape((ITERS, M_use, 1))

            if M_use > 1:
                cgz   = _N.zeros((ITERS, nSpks), dtype=_N.uint8)
            else:
                cgz   = _N.ones((ITERS, nSpks), dtype=_N.uint8)
                cls_str_ind[0] = 0
                cls_str_ind[1] = nSpks
                v_sts = Asts + t0
                clstszs[0, 0] = nSpks
            oo.cgz=cgz

            ########### stochasticAllocation variables
            xAS  = x[Asts + t0]   #  position @ spikes.  creates new copy
            mAS  = fit_mks[Asts + t0]   #  position @ spikes
            econt = _N.empty((M_use, nSpks))
            rat   = _N.zeros((M_use+1, nSpks))
            qdrMKS = _N.empty((M_use, nSpks))
            qdrSPC = _N.empty((M_use, nSpks))
            exp_arg= _N.empty((M_use, nSpks))
            ################################  GIBBS ITERS ITERS ITERS

            ttA = _tm.time()


            l0_a_is0    = _N.where(_l0_a == 0)[0]
            l0_a_Init   = _N.where(_l0_a >  0)[0]
            b_l0_a_is0  = len(l0_a_is0) > 0
            q2_a_is_m1  = _N.where(_q2_a == -1)[0]
            q2_a_Init   = _N.where(_q2_a > 0)[0]
            b_q2_a_is_m1= len(q2_a_is_m1) > 0

            _Dl0_a = _N.empty(M_use);            _Dl0_B = _N.empty(M_use)
            _Dq2_a = _N.empty(M_use);            _Dq2_B = _N.empty(M_use)

            iiq2 = 1./q2
            iiq2r= iiq2.reshape((M_use, 1))

            mcs = _N.empty((M_use, K))   # cluster sample means
            mcsT = _N.empty((M_use, K))   # cluster sample means
            outs1 = _N.empty((M_use, K))
            outs2 = _N.empty((M_use, K))

            BLK        = 1000
            iterBLOCKs = ITERS//BLK   # new in python 3  //  int/int = int

            #q2[:] = _N.random.rand(M_use)#100#_N.random.rand(M_use)
            #fb[:]   = _N.random.randn(M_use)



            inv_sum_sd2s_x = _N.empty((M_use, totalpcs))
            nrm_x          = _N.empty((M_use, totalpcs))
            diff2_x        = _N.empty((M_use, totalpcs))
            

            _cdfs.nrm_x(totalpcs, M_use, inv_sum_sd2s_x, nrm_x, q2, ap_sd2s_x)
            _cdfs.diff2_x(totalpcs, M_use, diff2_x, f, ap_mn_x)

            goback = 500

            #  B' / (a' - 1) = MODE   #keep mode the same after discount
            if (epc > 0) and oo.adapt:
                #  hyperparameter for q2
                _md_nd= _q2_B / (_q2_a + 1)
                _Dq2_a = _q2_a * _N.exp(-DT/tau_q2)
                _Dq2_B = _md_nd * (_Dq2_a + 1)

                #  hyperparameter for f
                _f_q2pr = _f_q2 + f_q2_rate * DT

                #  hyperparameter for l0                
                _mn_nd= _l0_a / _l0_B
                #  variance is a/b^2
                #  a/2 / B/2    variance is a/2 / B^2/4 = 2a^2 / B^2  
                #  variance increases by 2

                _Dl0_a = _l0_a * _N.exp(-DT/tau_l0)
                _Dl0_B = _Dl0_a / _mn_nd
            else:
                _Dq2_a = _q2_a
                _Dq2_B = _q2_B

                _Dl0_a = _l0_a
                _Dl0_B = _l0_B    #  = _l0_a / (f0 * _N.sqrt(2pi q2))
                #  make it so that

                _f_q2pr = _f_q2
            

            # if epc > ep1:
            #     print("..........................")
            #     print(_Sg_PSI)
            #     print("..........................")                
            #     print(_Sg_nu)
            #     print("..........................")
            oo.t1_0 = _tm.time()

            max_so_far = _N.empty(nSpks)
            
            for itrB in range(iterBLOCKs):
                #print("-------tetr %(t)d  itr  %(i)d" % {"i" : itrB*BLK, "t" : oo.tetr})
                tEnterBLK = _tm.time()
                
                for itr in range(itrB*BLK, (itrB+1)*BLK):
                    #ttsw1 = _tm.time()
                    iSg = _N.linalg.inv(Sg)
                    #ttsw2 = _tm.time()

                    if M_use > 1:
                        _sA.stochasticAssignment(epc, itr, M_use, K, l0, f, q2, u, Sg, iSg, _f_u, _u_u, _f_q2, pr_u_Sg, nSpks, t0, mAS, xAS, rat, econt, max_so_far, cgz, qdrSPC, qdrMKS, exp_arg, freeClstr, clstszs[itr])

                        _fm.cluster_bounds2_cgz(clstszs[itr], Asts, cls_str_ind, v_sts, cgz[itr], t0, M_use, nSpks)    # _fm.cluser_bounds provides no improvement
                                            
                        #print(str(clstszs[itr]))

                    #ttsw3 = _tm.time()
                    ###############
                    ###############     u
                    ###############
                    #if itr > 8900:
                    
                    _N.copyto(u_Sg_, _N.linalg.inv(i_pr_u_Sg + clstszs_rr[itr]*iSg))
                    _fm.find_mcs(clstszs[itr], v_sts, cls_str_ind, fit_mks, mcs, M_use, K)

                    _fm.multiple_mat_dot_v(i_pr_u_Sg, _u_u, outs1, M_use, K)
                    _fm.multiple_mat_dot_v(iSg, mcs, outs2, M_use, K)
                    _fm.multiple_mat_dot_v(u_Sg_, outs1 + clstszs_r[itr]*outs2, u_u_, M_use, K)

                    #ttsw4 = _tm.time()
                    ucmvnrms= _N.random.randn(M_use, K)

                    try:
                        C       = _N.linalg.cholesky(u_Sg_)
                    except _N.linalg.linalg.LinAlgError:
                        dmp = open("cholesky_tet%(t)d_ep%(e)d_itr%(i)d.dmp" % {"t" : oo.tetr, "i" : itr, "e" : epc}, "wb")
                        pickle.dump([u_Sg_, i_pr_u_Sg, clstszs[itr], Sg, iSg, pr_u_Sg, _u_u], dmp, -1)
                        dmp.close()

                    #  _u_Sg
                        raise
                    u       = _N.einsum("njk,nk->nj", C, ucmvnrms) + u_u_
                    smp_mk_prms[oo.ky_p_u][:, itr] = u  # dim of u wrong
                    smp_mk_hyps[oo.ky_h_u_u][:, itr] = u_u_
                    smp_mk_hyps[oo.ky_h_u_Sg][:, itr] = u_Sg_
                    
                    #ttsw5 = _tm.time()
                    ###############
                    ###############  Conditional f
                    ###############

                    m_rnds = _N.random.rand(M_use)

                    _cdfs.smp_f(itr, M_use, xt0t1, clstszs[itr], cls_str_ind, 
                                   v_sts, t0, l0,
                                   totalpcs, ap_Ns, ap_mn_x,
                                   diff2_x, nrm_x, inv_sum_sd2s_x,
                                   _f_u, _f_q2pr, m_rnds, f, q2, ignore_space)

                    #_cdfs.diff2_x(totalpcs, M_use, diff2_x, f, ap_mn_x)
                    
                    #f   = _N.array([6.33])
                    smp_sp_prms[oo.ky_p_f, :, itr] = f

                    #if itr > 8900:
                    #print("I itr %d" % itr)                    
                    
                    ##############
                    ##############  VARIANCE, COVARIANCE
                    ##############
                    #ttsw6 = _tm.time()
                    Sg_nu_ = _Sg_nu + clstszs[itr]# - (K-1)  #

                    _fm.Sg_PSI(cls_str_ind, clstszs[itr], v_sts, fit_mks, _Sg_PSI, Sg_PSI_, u, M_use, K)                    
                    #ttsw7 = _tm.time()                        
                    Sg = _iw.invwishartrand_multi_tempM(M_use, K, Sg_nu_, Sg_PSI_)
                    #ttsw8 = _tm.time()
                    
                    smp_mk_prms[oo.ky_p_Sg][:, itr] = Sg
                    smp_mk_hyps[oo.ky_h_Sg_nu][:, itr] = Sg_nu_
                    smp_mk_hyps[oo.ky_h_Sg_PSI][:, itr] = Sg_PSI_


                    ##############
                    ##############  SAMPLE SPATIAL VARIANCE
                    ##############
                    #  B' / (a' - 1) = MODE   #keep mode the same after discount

                    #ttsw9 = _tm.time()                    
                    m_rnds = _N.random.rand(M_use)
                    #  B' = MODE * (a' - 1)

                    _cdfs.smp_q2(itr, M_use, xt0t1, clstszs[itr], cls_str_ind, 
                                 v_sts, t0, l0,
                                 totalpcs, ap_Ns, ap_sd2s_x,
                                 diff2_x, nrm_x, inv_sum_sd2s_x,
                                 _Dq2_a, _Dq2_B, m_rnds, f, q2, ignore_space)

                    smp_sp_prms[oo.ky_p_q2, :, itr]   = q2

                    #ttsw10 = _tm.time()


                    ###############
                    ###############  CONDITIONAL l0
                    ###############
                    #  _ss.gamma.rvs.  uses k, theta  k is 1/B (B is our thing)
                    #_cdfs.l0_spatial(M_use, oo.dt, f, q2, l0_exp_hist)
                    _cdfs.l0_spatial(M_use, totalpcs, oo.dt, ap_Ns, nrm_x, diff2_x, inv_sum_sd2s_x, l0_exp_px_apprx)
                    

                    BL  = l0_exp_px_apprx    #  dim M
                    
                    aL  = clstszs[itr] + 1
                    l0_a_ = aL + _Dl0_a
                    l0_B_ = BL + _Dl0_B

                    l0 = _ss.gamma.rvs(l0_a_, scale=(1/l0_B_), size=M_use)  #  check                    
                    """
                    try:   #  if there is no prior, if a cluster 
                        l0 = _ss.gamma.rvs(l0_a_, scale=(1/l0_B_), size=M_use)  #  check
                        #print("pk firing")
                        #print(l0  / _N.sqrt(twpi*q2))
                        #l0 = _N.array([900.])
                    except ValueError:
                        print("problem with l0    %d" % itr)
                        print(l0_exp_hist)
                        print(l0_a_)
                        print(l0_B_)
                        raise
                    """
                    smp_sp_prms[oo.ky_p_l0, :, itr] = l0

                    #ttsw11 = _tm.time()
                    # print("#timing start")
                    # print("nt+= 1")
                    # print("t2t1+=%.4e" % (#ttsw2-#ttsw1))
                    # print("t3t2+=%.4e" % (#ttsw3-#ttsw2))
                    # print("t4t3+=%.4e" % (#ttsw4-#ttsw3))
                    # print("t5t4+=%.4e" % (#ttsw5-#ttsw4))
                    # print("t6t5+=%.4e" % (#ttsw6-#ttsw5))
                    # print("t7t6+=%.4e" % (#ttsw7-#ttsw6))  # slow
                    # print("t8t7+=%.4e" % (#ttsw8-#ttsw7))
                    # print("t9t8+=%.4e" % (#ttsw9-#ttsw8))
                    # print("t10t9+=%.4e" % (#ttsw10-#ttsw9))
                    # print("t11t10+=%.4e" % (#ttsw11-#ttsw10))
                    # print("#timing end  %.5f" % (#ttsw10-#ttsw1))

                    #if itr > 8900:
                    #    print("done itr %d" % itr)                    
                tFinishBLK = _tm.time()
                print("FINISH BLCK %(tet)d   Last 1000 iters:  %(tm).2f" % {"tet" : oo.tetr, "tm" : (tFinishBLK - tEnterBLK)})                
                stop_Gibbs = False
                cond1 = ((epc == 0) and (itr >= 6000))
                cond2 = ((epc > 0) and (itr >= 4000))


                if (global_stop_condition and (cond1 or cond2)):
                    print("I SHOULD NEVER BE HERE")
                    tttt1 = _tm.time()
                    stop_Gibbs = _pU.stop_Gibbs_cgz(itr, M_use, nSpks, smp_sp_prms, clstszs)
                    tttt2 = _tm.time()
                    print("tet %(tet)d   stop_Gibbs:  %(tm).2f" % {"tet" : oo.tetr, "tm" : (tttt2-tttt1)})
                    
                if stop_Gibbs:
                    print(global_stop_condition)
                    print("!!!!!  tetr %(tet)d   stop Gibbs at %(itr)d" % {"tet" : oo.tetr, "itr" : itr})
                    goback = 3000
                    break

                

            ttB = _tm.time()
            print(ttB-ttA)
            oo.t1_1 = _tm.time()                                    

            t11 = _tm.time()
            gAMxMu.finish_epoch2_cgz(oo, nSpks, epc, itr+1, clstszs, l0, f, q2, u, Sg, _f_u, _f_q2, _q2_a, _q2_B, _l0_a, _l0_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, smp_sp_prms, smp_mk_prms, smp_mk_hyps, freeClstr, M_use, K, priors, global_stop_condition, goback//3)            
            #  _l0_a is a copy of a subset of _l0_a_M
            #  we need to copy back the values _l0_a back into _l0_a_M
            t22 = _tm.time()
            gAMxMu.contiguous_inuse_cgz(M_use, M_max, K, freeClstr, l0, f, q2, u, Sg, _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, smp_sp_prms, smp_mk_prms, oo.sp_prmPstMd, oo.mk_prmPstMd, cgz, priors)
            t33 = _tm.time()
            gAMxMu.copy_back_params(M_use, l0, f, q2, u, Sg, M_max, l0_M, f_M, q2_M, u_M, Sg_M)
            t44 = _tm.time()            
            gAMxMu.copy_back_hyp_params(M_use, _l0_a, _l0_B, _f_u, _f_q2, _q2_a, _q2_B, _u_u, _u_Sg, _Sg_nu, _Sg_PSI, M_max, _l0_a_M, _l0_B_M, _f_u_M, _f_q2_M, _q2_a_M, _q2_B_M, _u_u_M, _u_Sg_M, _Sg_nu_M, _Sg_PSI_M)

            ########  NOW WE NEED TO DO THINGS FOR PARAMS, HYP PARAMS
            t55 = _tm.time()
            print("22-11 %.3f" % (t22-t11))
            print("33-22 %.3f" % (t33-t22))
            print("44-33 %.3f" % (t44-t33))
            print("55-44 %.3f" % (t55-t44))
            oo.t0_1 = _tm.time()                                    
            

            #  MAP of nzclstr
            if saveSamps:
                pcklme["smp_sp_prms"] = smp_sp_prms[:, 0:itr+1]
                #pcklme["smp_mk_prms"] = [smp_mk_prms[0][:, 0:itr+1], smp_mk_prms[1][:, :, 0:itr+1]]
                pcklme["smp_mk_prms"] = [smp_mk_prms[0][0:itr+1], smp_mk_prms[1][0:itr+1]]
            pcklme["sp_prmPstMd"] = oo.sp_prmPstMd
            pcklme["mk_prmPstMd"] = oo.mk_prmPstMd
            pcklme["intvs"]       = oo.intvs
            if saveOcc:
                pcklme["occ"]         = cgz[0:itr+1]
                pcklme["freeClstr"]           = freeClstr  #  next time
            pcklme["nz_pth"]         = nz_pth
            pcklme["M"]           = M_use
            pcklme["t0_0"]        = oo.t0_0
            pcklme["t1_0"]        = oo.t1_0
            pcklme["t1_1"]        = oo.t1_1
            pcklme["t0_1"]        = oo.t0_1

            print(resFN("posteriors_%d.dmp" % epc, dir=oo.outdir, __JIFIResultDir__=oo.__JIFIResultDir__))
            dmp = open(resFN("posteriors_%d.dmp" % epc, dir=oo.outdir, __JIFIResultDir__=oo.__JIFIResultDir__), "wb")
            pickle.dump(pcklme, dmp, -1)
            dmp.close()

            fp = open(resFN("posteriors_%d.txt" % epc, dir=oo.outdir, __JIFIResultDir__=oo.__JIFIResultDir__), "w")

            for m in range(M_use):
                fp.write("m %d\n" % m)
                fp.write("l0 %(l0).3f   f %(f).3f   sd %(sd).3f\n" % {"l0" : oo.sp_prmPstMd[3*m], "f" : oo.sp_prmPstMd[3*m+1], "sd" : _N.sqrt(oo.sp_prmPstMd[3*m+2])})
                fp.write("u [%(1).2f, %(2).2f, %(3).2f, %(4).2f]\n\n" % {"1" : oo.mk_prmPstMd[0][m, 0], "2" : oo.mk_prmPstMd[0][m, 1], "3" : oo.mk_prmPstMd[0][m, 2], "4" : oo.mk_prmPstMd[0][m, 3]})
                
                fp.write("cov [%(1).2f, %(2).2f, %(3).2f, %(4).2f]\n" % {"1" : oo.mk_prmPstMd[1][m, 0, 0], "2" : oo.mk_prmPstMd[1][m, 0, 1], "3" : oo.mk_prmPstMd[1][m, 0, 2], "4" : oo.mk_prmPstMd[1][m, 0, 3]})
                fp.write("cov [%(1).2f, %(2).2f, %(3).2f, %(4).2f]\n" % {"1" : oo.mk_prmPstMd[1][m, 1, 0], "2" : oo.mk_prmPstMd[1][m, 1, 1], "3" : oo.mk_prmPstMd[1][m, 1, 2], "4" : oo.mk_prmPstMd[1][m, 1, 3]})
                fp.write("cov [%(1).2f, %(2).2f, %(3).2f, %(4).2f]\n" % {"1" : oo.mk_prmPstMd[1][m, 2, 0], "2" : oo.mk_prmPstMd[1][m, 2, 1], "3" : oo.mk_prmPstMd[1][m, 2, 2], "4" : oo.mk_prmPstMd[1][m, 2, 3]})
                fp.write("cov [%(1).2f, %(2).2f, %(3).2f, %(4).2f]\n" % {"1" : oo.mk_prmPstMd[1][m, 3, 0], "2" : oo.mk_prmPstMd[1][m, 3, 1], "3" : oo.mk_prmPstMd[1][m, 3, 2], "4" : oo.mk_prmPstMd[1][m, 3, 3]})                


            fp.close()

        print("DONE WITH GIBBS GIBBS 1d  %(tet)d   %(save)s" % {"tet" : oo.tetr, "save" : oo.outdir})            
