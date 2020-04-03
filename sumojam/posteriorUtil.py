import numpy as _N
from filter import gauKer#, contiguous_pack2
import matplotlib.pyplot as _plt
import scipy.special as _ssp
import scipy.optimize as _scop
import scipy.stats as _ss
#from scipy.spatial.distance import cdist, euclidean

_GAMMA         = 0
_INV_GAMMA     = 1

def find_good_clstrs_and_stationary_from(M, smps, spcdim=1):
    #  treat narrow and wide cluster differently because the correlation
    #  timescale of sequential samples are quite different between slow
    #  and wide clusers
    if spcdim == 1:
        frm_narrow = stationary_from_Z_bckwd(smps, blksz=200)
        frm_wide   = stationary_from_Z_bckwd(smps, blksz=500)
    else:
        frm_narrow = stationary_from_Z_bckwd_2d(smps, blksz=200)
        frm_wide   = stationary_from_Z_bckwd_2d(smps, blksz=500)

    ITERS      = smps.shape[1]

    frms       = _N.empty(M, dtype=_N.int)


    if spcdim == 1:  # smp_sp_prms = _N.zeros((3, M_use, ITERS))  
        q2_mdn     = _N.median(smps[2, :, ITERS-1000:], axis=1)

        wd_clstrs  = _N.where(q2_mdn > 9)[0]
        nrw_clstrs  = _N.where(q2_mdn <= 9)[0]    
    else:
        q2x_mdn     = _N.median(smps[3, ITERS-1000:], axis=0)
        q2y_mdn     = _N.median(smps[4, ITERS-1000:], axis=0)

        wd_clstrs  = _N.where((q2x_mdn > 9) | (q2y_mdn > 9))[0]
        nrw_clstrs  = _N.where((q2x_mdn <= 9) | (q2y_mdn <= 9))[0]    

    print(frms.shape)
    print(frm_narrow.shape)
    print(nrw_clstrs.shape)
    print(wd_clstrs.shape)
    frms[nrw_clstrs] = frm_narrow[nrw_clstrs]
    frms[wd_clstrs] = frm_wide[wd_clstrs]

    return frms

"""    
def stationary_from_Z_bckwd(smps, blksz=200):
    #  Detect when stationarity reached in Gibbs sampling
    #  Also, detect whether cluster is switching between local extremas
    #
    SMPS, M   = smps.shape[1:]

    wins         = SMPS/blksz
    wins_m1      = wins - 1

    frms      = _N.empty(M, dtype=_N.int)  # current start of stationarity

    rshpd     = smps.reshape((3, wins, blksz, M))
    mrshpd    = _N.mean(rshpd, axis=2)   #  3 x wins_m1+1 x M
    sdrshpd   = _N.std(rshpd, axis=2)

    mLst                =         mrshpd[:, wins_m1].reshape(3, 1, M)
    sdLst               =         sdrshpd[:, wins_m1].reshape(3, 1, M)
    sdNLst               =         sdrshpd[:, 0:-1].reshape(3, wins_m1, M)

    zL                =         (mrshpd[:, 0:-1] - mLst)/sdLst
    zNL               =         (mrshpd[:, 0:-1] - mLst)/sdNLst

    #  mn, std in each win
    #  u1=0.3, sd1=0.9      u2=0.4, sd2=0.8
    #  (u2-u1)/sd1  

    #  detect sudden changes

    for m in xrange(M):
        win1stFound=(wins_m1-1)*blksz
        sameDist = 0
        i = 0

        thisWinSame     = False
        lastWinSame     = 0#wins_m1

        #  want 3 consecutive windows where distribution looks different
        while (sameDist <= 3) and (i < wins_m1-1):
            i += 1
            it0 = i*blksz
            it1 = (i+1)*blksz

            thisWinSame = 0

            for d in xrange(3):
                if ((zL[d, i, m] < 0.75) and (zL[d, i, m] > -0.75)) and \
                   ((zNL[d, i, m] < 0.75) and (zNL[d, i, m] > -0.75)):
                    
                    thisWinSame += 1

            if thisWinSame == 3:
                if sameDist == 0:
                    win1stFound = it0
                lastWinSame = i

                sameDist += 1

            if (i - lastWinSame > 1) and (sameDist <= 3):
                #print "reset  %d" % i
                sameDist = 0   #  reset
                win1stFound = (wins_m1-1)*blksz

        frms[m] = win1stFound

    return frms+blksz
"""

def stationary_from_Z_bckwd(smps, blksz=200, spcdim=1):
    #  Detect when stationarity reached in Gibbs sampling
    #  Also, detect whether cluster is switching between local extremas
    #
    M, SMPS   = smps.shape[1:]   #  smp_sp_prms = _N.zeros((3, ITERS, M_use))  

    wins         = SMPS//blksz   # 3.7
    wins_m1      = wins - 1

    frms      = _N.empty(M, dtype=_N.int)  # current start of stationarity

    if spcdim == 1:
        reparam     = _N.empty((2, M, SMPS))   #  reparameterized
        reparam[0]  = smps[1]
        reparam[1]  = smps[0] / _N.sqrt(smps[2])

        rshpd     = reparam.reshape((2, wins, blksz, M))
    else:
        reparam     = _N.empty((3, M, SMPS))   #  reparameterized
        reparam[0]  = smps[1]  # fx
        reparam[1]  = smps[2]  # fy
        reparam[2]  = smps[0] / _N.sqrt(smps[3]*smps[4])  #  

        rshpd     = reparam.reshape((3, wins, blksz, M))

    mrshpd    = _N.median(rshpd, axis=2)   #  2 x wins_m1+1 x m
    sdrshpd   = _N.std(rshpd, axis=2)

    mLst                =         mrshpd[:, wins_m1].reshape(2, 1, M)
    sdLst               =         sdrshpd[:, wins_m1].reshape(2, 1, M)
    sdNLst               =         sdrshpd[:, 0:-1].reshape(2, wins_m1, M)

    zL                =         (mrshpd[:, 0:-1] - mLst)/sdLst
    zNL               =         (mrshpd[:, 0:-1] - mLst)/sdNLst

    #  mn, std in each win
    #  u1=0.3, sd1=0.9      u2=0.4, sd2=0.8
    #  (u2-u1)/sd1  

    #  detect sudden changes

    for m in range(M):
        win1stFound=(wins_m1-1)*blksz
        sameDist = 0
        i = 0

        thisWinSame     = False
        lastWinSame     = 0#wins_m1

        #  want 3 consecutive windows where distribution looks different
        while (sameDist <= 2) and (i < wins_m1-1):
            i += 1
            it0 = i*blksz
            it1 = (i+1)*blksz

            thisWinSame = 0

            for d in range(2):
                if ((zL[d, i, m] < 0.75) and (zL[d, i, m] > -0.75)) and \
                   ((zNL[d, i, m] < 0.75) and (zNL[d, i, m] > -0.75)):
                    
                    thisWinSame += 1

            if thisWinSame == 2:
                if sameDist == 0:
                    win1stFound = it0
                lastWinSame = i

                sameDist += 1

            if (i - lastWinSame > 1) and (sameDist <= 2):
                #print "reset  %d" % i
                sameDist = 0   #  reset
                win1stFound = (wins_m1-1)*blksz

        frms[m] = win1stFound

    return frms+blksz

def stationary_from_Z_bckwd_2d(smps, blksz=200):
    #  Detect when stationarity reached in Gibbs sampling
    #  Also, detect whether cluster is switching between local extremas
    #
    SMPS, M   = smps.shape[1:]   #  smp_sp_prms = _N.zeros((3, ITERS, M_use))  

    wins         = SMPS//blksz   # python3
    wins_m1      = wins - 1

    frms      = _N.empty(M, dtype=_N.int)  # current start of stationarity

    reparam     = _N.empty((3, SMPS, M))   #  reparameterized
    reparam[0]  = smps[1]  # fx
    reparam[1]  = smps[2]  # fy
    reparam[2]  = smps[0] / _N.sqrt(smps[3]*smps[4])  #  

    rshpd     = reparam.reshape((3, wins, blksz, M))

    mrshpd    = _N.median(rshpd, axis=2)   #  2 x wins_m1+1 x m
    sdrshpd   = _N.std(rshpd, axis=2)

    mLst                =         mrshpd[:, wins_m1].reshape(3, 1, M)
    sdLst               =         sdrshpd[:, wins_m1].reshape(3, 1, M)
    sdNLst               =         sdrshpd[:, 0:-1].reshape(3, wins_m1, M)

    zL                =         (mrshpd[:, 0:-1] - mLst)/sdLst
    zNL               =         (mrshpd[:, 0:-1] - mLst)/sdNLst

    #  mn, std in each win
    #  u1=0.3, sd1=0.9      u2=0.4, sd2=0.8
    #  (u2-u1)/sd1  

    #  detect sudden changes

    for m in range(M):
        win1stFound=(wins_m1-1)*blksz
        sameDist = 0
        i = 0

        thisWinSame     = False
        lastWinSame     = 0#wins_m1

        #  want 3 consecutive windows where distribution looks different
        while (sameDist <= 2) and (i < wins_m1-1):
            i += 1
            it0 = i*blksz
            it1 = (i+1)*blksz

            thisWinSame = 0

            for d in range(3):
                if ((zL[d, i, m] < 0.75) and (zL[d, i, m] > -0.75)) and \
                   ((zNL[d, i, m] < 0.75) and (zNL[d, i, m] > -0.75)):
                    
                    thisWinSame += 1

            if thisWinSame == 3:
                if sameDist == 0:
                    win1stFound = it0
                lastWinSame = i

                sameDist += 1

            if (i - lastWinSame > 1) and (sameDist <= 2):
                #print "reset  %d" % i
                sameDist = 0   #  reset
                win1stFound = (wins_m1-1)*blksz

        frms[m] = win1stFound

    return frms+blksz






def stop_Gibbs(currIT, M, N, smp_sp_prms, gz):
    thin  = 10
    BLKsz = 500//thin  #
    statWin = 3000//thin   #  a window over which we want stationarity
    BLKS      = statWin // BLKsz

    occs = _N.sum(gz, axis=1)
    smps_sp_prms = _N.array(smp_sp_prms[:, :, currIT-statWin*thin:currIT:thin])
    sp_prms      = _N.empty((2, M, statWin))
    sp_prms[0]   = smps_sp_prms[1]
    sp_prms[1]   = smps_sp_prms[0] / _N.sqrt(smps_sp_prms[2])
    #smps_mk_prms = _N.array(lm["smp_mk_prms"][0][:, currIT-statWin*thin:currIT:thin])
    occs_pcs     = _N.array(occs[currIT-statWin*thin:currIT:thin])

    r            = 6
    n            = BLKsz
    
    #####  mean of latest samples
    mf_sp     = _N.mean(sp_prms[:, :, statWin-500//thin:statWin], axis=2, keepdims=True)
    wgts      =    _N.mean(occs[statWin-500//thin:statWin], axis=0, keepdims=True)
    wgts      *= M/N
    wgts2     =  _N.empty((2, M, 1))
    wgts2[0, :, 0]  = wgts
    wgts2[1, :, 0]  = wgts

    alp      = 0.01  #  
    #####  difference from mean of final bit

    sp_y_ij  = _N.mean(_N.abs(sp_prms - mf_sp) * wgts2, axis=1).reshape((2, BLKS, BLKsz))   #  clusters weighted
    #sp_y_ij[1, 0, 0] = 10000
    sp_y_dd  = _N.mean(_N.mean(sp_y_ij, axis=2, keepdims=True), axis=1, keepdims=True)
    sp_y_id  = _N.mean(sp_y_ij, axis=2, keepdims=True)

    SSE      = _N.sum(_N.sum((sp_y_ij - sp_y_id)**2, axis=2), axis=1)
    SSR      = n*_N.sum((sp_y_id - sp_y_dd)**2, axis=1)
    MSE      = SSE / ((n-1)*r)
    MSR      = SSR / (r-1)
    
    Fst      = MSR.T[0] / MSE
    Fthr     = _ss.f.ppf(1-alp, r-1, (n-1)*r)

    if (Fst[0] < Fthr) & (Fst[1] < Fthr):
        return True
    return False


def stop_Gibbs_cgz(currIT, M, N, smp_sp_prms, occs, spcdim=1):
    thin  = 10
    BLKsz = 500//thin  #
    statWin = 3000//thin   #  a window over which we want stationarity
    BLKS      = statWin // BLKsz

    #occs = _N.sum(gz, axis=1)
    smps_sp_prms = _N.array(smp_sp_prms[:, :, currIT-statWin*thin:currIT:thin])
    sp_prms   = None

    if spcdim == 1:
        sp_prms      = _N.empty((2, M, statWin))        
        sp_prms[0]   = smps_sp_prms[1]
        sp_prms[1]   = smps_sp_prms[0] / _N.sqrt(smps_sp_prms[2])
    else:
        sp_prms      = _N.empty((3, M, statWin))
        print("!!!!!!!!!!!!!!!!!!!!!!!!")
        print(smp_sp_prms.shape)
        print(sp_prms.shape)        
        sp_prms[0]   = smps_sp_prms[1]
        sp_prms[1]   = smps_sp_prms[2]
        sp_prms[2]   = smps_sp_prms[0] / _N.sqrt(smps_sp_prms[3] * smps_sp_prms[4])
        
    #smps_mk_prms = _N.array(lm["smp_mk_prms"][0][:, currIT-statWin*thin:currIT:thin])
    occs_pcs     = _N.array(occs[currIT-statWin*thin:currIT:thin])

    r            = 6
    n            = BLKsz
    
    #####  mean of latest samples
    mf_sp     = _N.mean(sp_prms[:, :, statWin-500//thin:statWin], axis=2, keepdims=True)
    wgts      =    _N.mean(occs[statWin-500//thin:statWin], axis=0, keepdims=True)
    print("wgts shape")
    print(wgts.shape)
    print("mf_sp shape")
    print(mf_sp.shape)
    print("sp_prms shape")
    print(sp_prms.shape)

    alp      = 0.01  #      
    wgts      *= M/N
    if spcdim == 1:
        wgts2     =  _N.empty((2, M, 1))
        wgts2[0, :, 0]  = wgts
        wgts2[1, :, 0]  = wgts
        sp_y_ij  = _N.mean(_N.abs(sp_prms - mf_sp) * wgts2, axis=1).reshape((2, BLKS, BLKsz))   #  clusters weighted        
    else:
        wgts2     =  _N.empty((3, M, 1))
        wgts2[0, :, 0]  = wgts
        wgts2[1, :, 0]  = wgts
        wgts2[2, :, 0]  = wgts
        sp_y_ij  = _N.mean(_N.abs(sp_prms - mf_sp) * wgts2, axis=1).reshape((3, BLKS, BLKsz))   #  clusters weighted

    #####  difference from mean of final bit

    #sp_y_ij[1, 0, 0] = 10000
    sp_y_dd  = _N.mean(_N.mean(sp_y_ij, axis=2, keepdims=True), axis=1, keepdims=True)
    sp_y_id  = _N.mean(sp_y_ij, axis=2, keepdims=True)

    SSE      = _N.sum(_N.sum((sp_y_ij - sp_y_id)**2, axis=2), axis=1)
    SSR      = n*_N.sum((sp_y_id - sp_y_dd)**2, axis=1)
    MSE      = SSE / ((n-1)*r)
    MSR      = SSR / (r-1)
    
    Fst      = MSR.T[0] / MSE
    Fthr     = _ss.f.ppf(1-alp, r-1, (n-1)*r)

    if (spcdim == 1) and (Fst[0] < Fthr) and (Fst[1] < Fthr):
        return True
    elif (spcdim == 2) and (Fst[0] < Fthr) and (Fst[1] < Fthr) & (Fst[2] < Fthr):
        return True
    
    return False


    
def stationary_from_Z(smps, blksz=200):
    M, SMPS   = smps.shape[1:]   #  smps shape changed

    wins      = SMPS//blksz - 1   # python3

    pvs       = _N.empty((M, 2, wins))
    ds        = _N.empty((M, 2, wins))
    frms      = _N.empty(M, dtype=_N.int)

    rshpd     = smps.reshape((3, wins+1, blksz, M))
    mrshpd    = _N.mean(rshpd, axis=2)
    sdrshpd   = _N.std(rshpd, axis=2)

    mLst                =         mrshpd[:, wins].reshape(3, 1, M)
    sdLst               =         sdrshpd[:, wins].reshape(3, 1, M)
    sdNLst               =         sdrshpd[:, 0:-1].reshape(3, wins, M)

    zL                =         (mrshpd[:, 0:-1] - mLst)/sdLst
    zNL               =         (mrshpd[:, 0:-1] - mLst)/sdNLst

    
    for m in range(M):
#        print("mmmmmmmmmmmmmmmmmmm  %d" % m)
        for d in range(3):
            diff_dist          =         _N.where((zL[d, :, m] > 0.75) | (zL[d, :, m] < -0.75))[0]
#            print(diff_dist)
                                           
        win1stFound=0
        diffDist = 0
        i = wins
        win1stDffrnt      = wins - 1
        lastWinDffrnt     = wins - 1
        thisWinDffrnt     = False

        while (diffDist <= 6) and (i > 0):
            i -= 1
            it0 = i*blksz
            it1 = (i+1)*blksz

            for d in range(3):
                if ((zL[d, i, m] > 0.75) or (zL[d, i, m] < -0.75)) or \
                   ((zNL[d, i, m] > 0.75) or (zNL[d, i, m] < -0.75)):
                    if diffDist == 0:
                        win1stFound = it0

                    lastWinDffrnt     = i                    
                    diffDist += 1
                    #print "%(i)d stats of block different than earlier  diffDist %(dD)d"  % {"i" : i, "dD" : diffDist}
            #print "lastWinDffrnt - i  %d" % (lastWinDffrnt - i)
            if (lastWinDffrnt - i > 1) and (diffDist <= 6):
                #print "reset  %d" % i
                diffDist = 0   #  reset

        frms[m] = win1stFound

    return frms+blksz

def MAPvalues2(epc, smp_prms, postMode, frms, ITERS, M, nprms, occ, l_trlsNearMAP, alltrials=False):

    for m in range(M):
        frm = frms[m]
        #fig = _plt.figure(figsize=(11, 4))
        trlsNearMAP = _N.arange(0, ITERS-frm)
        if alltrials:
            l_trlsNearMAP.append(_N.arange(ITERS))
        else:
            for ip in range(nprms):  # for each param
                col = nprms*m+ip      #  0, 3, 6  for lambda0
                #fig.add_subplot(1, nprms, ip+1)

                smps  = smp_prms[ip, frm:, m]
                #postMode[epc, col] = _N.mean(smps)
                postMode[epc, col] = _N.median(smps)
            if l_trlsNearMAP is not None:
                # trlsNearMAP for each params added to list
                l_trlsNearMAP.append(trlsNearMAP)  

def gam_inv_gam_dist_ML(smps, dist=_GAMMA, clstr=None):

    """
    The a, B hyperparameters for gamma or inverse gamma distributions
    """
    N = len(smps)

    s_x_ix = _N.sum(smps) if dist == _GAMMA else _N.sum(1./smps)

    if len(_N.where(smps == 0)[0]) > 0:
        print("0 found for cluster %(c)d   for dist %(d)s" % {"c" : clstr, "d" : ("gamma" if dist == _GAMMA else "inv gamma")})
        return None, None
    
    pm_s_logx = _N.sum(_N.log(smps)) 
    pm_s_logx *= 1 if dist == _GAMMA else -1

    mn  = _N.mean(smps)
    vr  = _N.std(smps)**2
    BBP = (mn / vr) if dist == _GAMMA else mn*(1 + (mn*mn)/vr)

    Bx  = _N.linspace(BBP/50, BBP*50, 1000)
    yx  = _N.empty(1000)
    iB  = -1
    
    for B in Bx:
        iB += 1
        P0 = _ssp.digamma(B/N * s_x_ix)
        yx[iB] = N*(_N.log(B) - P0) + pm_s_logx

    lst = _N.where((yx[:-1] >= 0) & (yx[1:] <= 0))[0]
    if len(lst) > 0:
        ib4 = lst[0]
        #  decreasing
        mslp  = (yx[ib4+1] - yx[ib4]) 
        rd    = (0-yx[ib4])  / mslp

        Bml = Bx[ib4] + rd*(Bx[ib4+1]-Bx[ib4])
        aml = (Bml/N)*s_x_ix   #  sum xi / N  = B/a
        return aml, Bml
    else:
        return None, None

def gam_inv_gam_dist_ML_v2(smps, dist=_GAMMA, clstr=None):

    """
    The a, B hyperparameters for gamma or inverse gamma distributions
    """
    N = len(smps)

    s_x_ix = _N.sum(smps) if dist == _GAMMA else _N.sum(1./smps)

    if len(_N.where(smps == 0)[0]) > 0:
        print("0 found for cluster %(c)d   for dist %(d)s" % {"c" : clstr, "d" : ("gamma" if dist == _GAMMA else "inv gamma")})
        return None, None
    
    pm_s_logx = _N.sum(_N.log(smps)) 
    pm_s_logx *= 1 if dist == _GAMMA else -1

    #  estimate range of alphas necessary 
    mn  = _N.mean(smps)
    vr  = _N.std(smps)**2
    aaP = (mn**2 / vr) if dist == _GAMMA else (mn**2 / vr) + 2

    alps  = _N.linspace(aaP / 10, aaP * 10, 1000)
    yx = N*(_N.log(N*alps/s_x_ix) - _ssp.digamma(alps)) + pm_s_logx
    
    lst = _N.where((yx[:-1] >= 0) & (yx[1:] <= 0))[0]
    
    if len(lst) > 0:
        ib4 = lst[0]
        #  decreasing
        mslp  = (yx[ib4+1] - yx[ib4]) 
        rd    = (0-yx[ib4])  / mslp

        aml = alps[ib4] + rd*(alps[ib4+1]-alps[ib4])
        Bml = (aml*N)/s_x_ix   #  sum xi / N  = B/a
        return aml, Bml
    else:
        return None, None

def idML_f(nu, N, K, sum_inv_iws, sum_log_det_iws):
    """
    pdf of inv wishart-distributed variables
    """
    hnu = nu*0.5
    return N*K*(_N.log(nu) - _N.log(2)) - N*_N.log(_N.linalg.det(sum_inv_iws/N)) - N*(_ssp.polygamma(0, hnu) + _ssp.polygamma(0, hnu - 0.5) + _ssp.polygamma(0, hnu - 1) + _ssp.polygamma(0, hnu - 1.5)) - sum_log_det_iws


def idML_dfdnu(nu, N, K, sum_inv_iws, sum_log_det_iws):
    """
    deriv of pdf of inv wishart-distributed variables wrt deg of freedom
    """
    hnu = nu*0.5
    return N*K/nu - 0.5*N*(_ssp.polygamma(1, hnu) + _ssp.polygamma(1, hnu - 0.5) + _ssp.polygamma(1, hnu - 1) + _ssp.polygamma(1, hnu - 1.5))
    

def iWish_dist_ML(smps, K, clstr=None):
    inv_iws = _N.linalg.inv(smps)
    sum_inv_iws = _N.sum(inv_iws, axis=0)
    sum_log_det_iws = _N.sum(_N.log(_N.linalg.det(smps)))
    SMPLS       = smps.shape[0]

    nu_ml = _scop.newton(idML_f, _N.array([K+1]), fprime=idML_dfdnu, args=(SMPLS, K, sum_inv_iws, sum_log_det_iws), maxiter=10000)[0]

    PSI_ml = _N.linalg.inv((1/(nu_ml*SMPLS))*sum_inv_iws)
    return nu_ml, PSI_ml


############################################
def GamML_f(a, N, sum_gam_smps, sum_log_gam_smps):
    """
    pdf of inv wishart-distributed variables
    """
    return N*_N.log(N*a/sum_gam_smps) - N*_ssp.polygamma(0, a) + sum_log_gam_smps

def GamML_dfda(a, N, sum_gam_smps, sum_log_gam_smps):
    """
    deriv of pdf of inv wishart-distributed variables wrt deg of freedom
    """
    return N/a - N*_ssp.polygamma(1, a)

def Gam_dist_ML(smps):
    SMPLS       = smps.shape[0]
    sum_gam_smps = _N.sum(smps)
    sum_log_gam_smps = _N.sum(_N.log(smps))

    a_ml = _scop.newton(GamML_f, _N.array([0.1]), fprime=GamML_dfda, args=(SMPLS, sum_gam_smps, sum_log_gam_smps), maxiter=10000)[0]

    B_ml = SMPLS*a_ml/sum_gam_smps
    return a_ml, B_ml

############################################
def IG_ML_f(a, N, inv_sum_IG_smps, sum_log_IG_smps):
    """
    pdf of inv wishart-distributed variables
    """
    return N*_N.log(N*a/inv_sum_IG_smps) - N*_ssp.polygamma(0, a) - sum_log_IG_smps

def IG_ML_dfda(a, N, inv_sum_IG_smps, sum_log_IG_smps):
    """
    deriv of pdf of inv wishart-distributed variables wrt deg of freedom
    """
    return N/a - N*_ssp.polygamma(1, a)

def IG_dist_ML(smps):
    SMPLS       = smps.shape[0]
    sum_inv_IG_smps = _N.sum(1./smps)
    sum_log_IG_smps = _N.sum(_N.log(smps))

    a_ml = _scop.newton(IG_ML_f, _N.array([0.1]), fprime=IG_ML_dfda, args=(SMPLS, sum_inv_IG_smps, sum_log_IG_smps), maxiter=10000)[0]

    B_ml = SMPLS*a_ml/sum_inv_IG_smps
    return a_ml, B_ml
