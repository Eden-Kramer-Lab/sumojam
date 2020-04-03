##  decode script
#import matplotlib
#matplotlib.use('Agg')
import jifimogs.mkdecoder as _mkd
import pickle as _pkl
import jifimogs.tools.plottools as mF
import os
import jifimogs.tools.decodeutil as _du
import jifimogs.tools.manypgs as _mpgs
import time as _tm
import shutil
import matplotlib.ticker as ticker
import numpy as _N
import matplotlib.pyplot as _plt
from jifimogs.devscripts.cmdlineargs import process_keyval_args
import sys

_plt.ioff()    
_KDE_RT     = True   #  REALTIME:  data+behavior to be decoded used to train

fmts        = ["eps"]
thnspks    = 1


##############################
anim       = "bond" # bn2
day        = "03"
ep         = "02"
"""
##############################
anim       = "frank" # bn2
day        = "06"
ep         = "02"

##############################
anim       = "frank" # bn2
day        = "02"
ep         = "02"

##############################
anim       = "bond"
day        = "10"
ep         = "04"

##############################
anim       = "10bn"
day        = None
ep         = None

anim       = "bc5"
day        = "01"
ep         = "01"
"""
mode        = 1   #  single is mode 0, all is mode 1
itvfn   = "itv8_1"
#itvfn    = "itv2_6"
itvfn   = "itv2_11"
#itvfn   = "itv10_1"
#itvfn   = "itv15_2"
#itvfn   = "itv2_8"
#itvfn   = "itv12_1"
itvfn   = "itv8_2"
#itvfn   = "itv9_1"

saveBin  = 1    #  decode @ 1ms, but we save it at lower resolution

lagfit     = 1   # normally set to 1.  But set larger if we want to use very old fit to decode
epchBlks   = 1

EPCHS =8
#EPCHS =2
K     = 4

#itvfn   = "itv2_5"

label     =  "52"    #  refers to

bUseNiC    = True   # nn-informative cluster

#datfn      = ["mdec%s" % outdir]
method     = _du._MOG

process_keyval_args(globals(), sys.argv[1:])
############################################################

if (day is not None) and (ep is not None):
    usemaze    = _mkd.mz_W if ((anim == "frank") or (anim == "bond")) else _mkd.mz_CRCL
    andyep = anim + day + ep
    if andyep == "bond1004":
        _tetslist = list(range(100, 123))
        _tetslist.pop(21)
        _tetslist.pop(16)
    elif andyep == "bond0302":
        _tetslist = list(range(100, 124))
        _tetslist.pop(22)
    elif andyep == "frank0202":
        _tetslist = list(range(100, 118))
        #_tetslist.pop(16)
    elif andyep == "frank0602":
        _tetslist = list(range(100, 120))
        _tetslist.pop(17)        
        _tetslist.pop(15)
    elif andyep == "noch0101":
        _tetslist = list(range(1, 11))
    elif andyep == "bich0101":
        _tetslist = list(range(11, 21))
    elif andyep[0:2] == "bc":
        _tetslist = list(range(1, 7))
    elif andyep[0:2] == "sm":
        _tetslist = list(range(1, 7))
    _tetslist        = _N.array(_tetslist)
    _tetslist  = _tetslist.reshape((1, _tetslist.shape[0]))

    tetslist     = _tetslist
    if mode == 0:
        tetslist = _tetslist.T
    dat_mode = _du._TETS
    tet_or_sim_list = tetslist
else:
    usemaze    = _mkd.mz_CRCL
    andyep     = anim
    dat_mode = _du._SIMS
    tet_or_sim_list = _N.arange(16, 17)



xLo   = -6;  xHi   = 6

skp_disp  = 100   # every 20 ms.
hw_rat    = (1./5)

bx=0.15
Bx=0.15
Bm=4.

iResolution =1
if iResolution==2:
    Nx = 241
elif iResolution==1:
    Nx = 121
else:
    Nx = 61

lmrotate=False
#for tets in tetslist:
#    if tets is not None:
for tet_or_sim in tet_or_sim_list:
    if dat_mode == _du._TETS:
        tets = tet_or_sim
        if mode == 0:
            print("!!!  tets:  %d  !!!!  " % tets)
        datfns, resdns, outdir, tetstr = _du.makefilenames(anim, day, ep, tets.tolist(), itvfn, label=label, ripple=False)
    else:
        tets = None
        datfns, resdns, outdir, tetstr = _du.makefilenames(anim+str(tet_or_sim), day, ep, None, itvfn, label=label, ripple=False)
        print("!!!  sim %s  !!!! " % (anim + str(tet_or_sim)))

    if method == _du._KDE:
        if _KDE_RT:
            outdirN = "%(d)s/decKDE_RT%(ts)s_%(lf)d_%(Bx).2f_%(Bm).2f_%(bx).2f" % {"d" : outdir, "ts" :tetstr, "lf" : lagfit, "Bx" : Bx, "Bm" : Bm, "bx" : bx}
        else:
            outdirN = "%(d)s/decKDE%(ts)s_%(lf)d" % {"d" : outdir, "ts" :tetstr, "lf" : lagfit}
    elif method == _du._MOG:
        outdirN = "%(d)s/decMoG%(ts)s_%(lf)d" % {"d" : outdir, "ts" :tetstr, "lf" : lagfit}
    else:
        outdirN = "%(d)s/decGT%(ts)s_%(lf)d" % {"d" : outdir, "ts" :tetstr, "lf" : lagfit}

    if not os.access(outdir, os.F_OK):
        os.mkdir(outdir)
    if not os.access(outdirN, os.F_OK):
        os.mkdir(outdirN)
    shutil.copyfile("decMoGvarM.py", "%s/decMoG.py" % outdirN)

    #0, 4, 6, 9, 14

    dd    = os.getenv("__JIFIDataDir__")
    _dat   = _N.loadtxt("%(dd)s/%(dfn)s.dat" % {"dd" : dd, "dfn" : datfns[0]})  # DATA/anim_mdec_dyep_tet.dat
    dat   = _dat[:, 0:2+K]

    intvs = _N.array(_N.loadtxt("%(dd)s/%(iv)s.dat" % {"dd" : dd, "iv" : itvfn})*dat.shape[0], dtype=_N.int)

    all_tet_prms = []  #  one for each tetrode
    idn = -1
    for dn in resdns:   #  1 result dir for each tetrode
        idn += 1   #  For GT case
        #dfn = "%(tet)s-%(itv)s" % {"tet" : nt, "itv" : itv_runN}
        if method == _du._KDE:
            chmins = None
            if os.access("%s/chmins.dat" % dn, os.F_OK):
                chmins = _N.loadtxt("%s/chmins.dat" % dn)
            lmrotate = False
        if method == _du._MOG:
            with open("%s/posteriors_0.dmp" % dn, "rb") as f:
                lm = _pkl.load(f)
                f.close()

            try:
                chmins = lm["chmins"]
            except KeyError:
                chmins = None
            prms_4_each_epoch = []
            for e in range(0, EPCHS-1):
                with open("%(d)s/posteriors_%(e)d.dmp" % {"d" : dn, "e" : e}, "rb") as f:
                    lm = _pkl.load(f)

                    f.close()

                M            = lm["M"]                    
                if "freeClstr" in lm:
                    inuse_clstr = _N.where(lm["freeClstr"] == False)[0]
                else:
                    inuse_clstr = _N.arange(M)
                    
                Muse         = len(inuse_clstr)-3
                Mwowonz      = M
                if bUseNiC and "smp_nz_l0" in lm:
                    Mwowonz  = M + 1

                spmk_us      = _N.zeros((Mwowonz, K+1))          
                spmk_covs    = _N.zeros((Mwowonz, K+1, K+1))
                spmk_l0s     = _N.zeros((Mwowonz, 1))

                if Mwowonz == M:
                    mkprms       = lm["mk_prmPstMd"]
                    #mkprms       = [_N.median(lm["smp_mk_prms"][0][:, 2000:], axis=1).T, _N.median(lm["smp_mk_prms"][1][:, :, 2000:], axis=2).T]
                    l0s          = lm["sp_prmPstMd"][::3]
                    fs           = lm["sp_prmPstMd"][1::3]
                    q2s          = lm["sp_prmPstMd"][2::3]
                    nanq2s       = _N.where(_N.isnan(q2s))   #  old version set q2s to nan
                    if len(nanq2s) > 0:
                        q2s[nanq2s]  = 10000.
                else:
                    mkprms       = [_N.empty(lm["mk_prmPstMd"][0].shape + _N.array([0, 1, 0])), _N.empty(lm["mk_prmPstMd"][1].shape + _N.array([0, 1, 0, 0]))]
                    mkprms[0][:, 0:M]    = lm["mk_prmPstMd"][0]
                    mkprms[1][:, 0:M]    = lm["mk_prmPstMd"][1]
                    # (epchs, M, K), (epchs, M, K, K)
                    l0s          = _N.empty(Mwowonz)
                    l0s[0:M]      = lm["sp_prmPstMd"][::3]
                    l0s[Muse:M]  = 0
                    ITERS        = lm["smp_nz_l0"].shape[0]
                    #l0s[M]        = _N.median(lm["smp_nz_l0"][int(ITERS*0.7):])
                    fs           = _N.empty(Mwowonz)
                    fs[0:M]      = lm["sp_prmPstMd"][1::3]
                    fs[M]        = lm["nz_fs"]
                    q2s          = _N.empty(Mwowonz)
                    q2s[0:M]     = lm["sp_prmPstMd"][2::3]
                    q2s[M]       = lm["nz_q2"]

                us           = mkprms[0]
                covs         = mkprms[1]
                # for m in xrange(M):
                #     if covs[m, K-1, K-1] > 10:
                #         covs[m, K-1, K-1] = 0.01   ##  really large variance last element.  Why?
                if Mwowonz > M:
                    us[M]    = lm["nz_u"]
                    covs[M]    = lm["nz_Sg"]
                try:
                    lmrotate = lm["rotate"]
                except KeyError:
                    lmrotate = False

                ix = _N.where(l0s <= 0)[0]

                spmk_l0s[:, 0] = l0s
                spmk_us[:, 0] = fs
                spmk_us[:, 1:] = us
                spmk_covs[:, 0, 0] = q2s
                spmk_covs[:, 1:, 1:] = covs

                if len(ix) > 0:
                    #print("l0s <= 0, before")
                    #print(l0s[ix])
                    l0s[ix] = 0
                    spmk_l0s[:, 0] = l0s
                    spmk_covs[ix, 1:, 1:] = _N.eye(K)  #  sometimes non
                    #print(_N.linalg.det(spmk_covs))

                prms_4_each_epoch.append([spmk_l0s, spmk_us, spmk_covs])
            all_tet_prms.append(prms_4_each_epoch)
        if method == _du._GT:
            with open("%(dd)s/%(dfn)s_prms.pkl" % {"dd" : dd, "dfn" : datfns[idn]}, "rb") as f:
                gtprms = _pkl.load(f)
                gt_Dt    = gtprms["intv"]   #  how often gtprms sampled
            kms = gtprms["km"]    
            Mgt  = max([item for sublist in kms for item in sublist]) + 1

            chmins = None

            gt_us      = _N.zeros((EPCHS, Mgt, K+1))          
            gt_covs    = _N.zeros((EPCHS, Mgt, K+1, K+1))
            gt_l0s     = _N.zeros((EPCHS, Mgt, 1))

            l = 0
            prms_4_each_epoch = []
            for epc in range(EPCHS-1):
                t0 = int(intvs[epc] / gt_Dt)  #  decode times
                t1 = int(intvs[epc+1] / gt_Dt)

                gt_l0s[epc, :, 0] = _N.mean(gtprms["l0"][:, t0:t1], axis=1)
                ###########
                j_fs   = _N.mean(gtprms["f"][:, t0:t1], axis=1)
                j_q2   = _N.mean(gtprms["sq2"][:, t0:t1], axis=1)
                j_us   = _N.mean(gtprms["u"][:, t0:t1], axis=1)
                j_covs = gtprms["covs"]

                ###########
                for cl in range(len(kms)):
                    pfis = kms[cl]   #  place fields of each cell
                    for ipf in pfis:
                        gt_us[epc, ipf, 0]  = j_fs[ipf, ]
                        gt_us[epc, ipf, 1:] = j_us[cl]

                        gt_covs[epc, ipf, 0, 0] = j_q2[ipf, ]
                        gt_covs[epc, ipf, 1:, 1:] = j_covs[cl]

                prms_4_each_epoch.append([gt_l0s[epc], gt_us[epc], gt_covs[epc]])
                #all_gt_prms[0].append([gt_l0s[epc], gt_us[epc], gt_covs[epc]])
            all_tet_prms.append(prms_4_each_epoch)

    ltets = 1 if (tets is None) else len(tets)
    mkd   = _mkd.mkdecoder(kde=(method == _du._KDE), Bx=Bx, Bm=Bm, bx=bx, K=K, nTets=ltets, mkfns=datfns, xLo=xLo, xHi=xHi, maze=usemaze, spdMult=0.1, ignorespks=False, chmins=chmins, Nx=Nx, rotate=lmrotate, t1=intvs[-1])

    offs  = (0-xLo)

    #  we should be adding another axis for tetrode
    silenceLklhds = _N.empty((mkd.Nx, EPCHS-1))


    ##################################  decoding here
    for epch in range((epchBlks-1)+lagfit, EPCHS, epchBlks):
        t0 = intvs[epch]   #  decode times
        t1 = intvs[epch+epchBlks]
        print("t0: %(t0)d     t1: %(t1)d" % {"t0" : t0, "t1" : t1})
        tt0 = _tm.time()
        if method == _du._MOG:
            mkd.decodeMoG(all_tet_prms, epch-lagfit, t0, t1)   # epch-1 means use fit from previous epoch
        if method == _du._KDE:
            if _KDE_RT:
                mkd.prepareDecKDE(0, t0)
            else:
                mkd.prepareDecKDE(0, t1)
            mkd.decodeKDE(t0, t1)
        if method == _du._GT:
            mkd.decodeMoG(all_tet_prms, epch-lagfit, t0, t1, noninformative_silence=False)
        tt1 = _tm.time()
        print("decode time:   %.3f" % (tt1-tt0))

        #  should do a for-loop here.
        silenceLklhds[:, epch-1] = _N.exp(-mkd.dt * mkd.Lam_MoMks[:, 0])

        #fig = _plt.figure(figsize=(13, 3.4))
        #ax  = fig.add_subplot(1, 1, 1)

        show_px = mkd.pX_Nm[t0:t1:skp_disp]
        imw = float(show_px.shape[0])
        imh = float(show_px.shape[1])

        print("t0: %(0)d   t1: %(1)d" % {"0" : t0, "1" : t1})
        #_plt.imshow(show_px.T**0.25, aspect=((imw/imh)*hw_rat), cmap=_plt.get_cmap("gist_yarg"))
        #_plt.clim(0, _N.max(show_px**0.25))   #  treats pure noise spikes correctly.  -  we expect flat posterior.  Without this, clim is (min, max) - so even if nearly flat, posterior appears non-flat, but this is because of the limits
        #_plt.plot(_N.linspace(t0-t0, int(_N.ceil(float(t1-t0)/skp_disp))-1, int(_N.ceil(float(t1-t0)/skp_disp)))*(skp_disp/1000.), (mkd.pos[t0:t1:skp_disp]+offs)/mkd.dxp, color="orange", lw=3.5)
        #_plt.plot(_N.linspace(t0-t0, int(_N.ceil(float(t1-t0)/skp_disp))-1, int(_N.ceil(float(t1-t0)/skp_disp))), (mkd.pos[t0:t1:skp_disp]+offs)/mkd.dxp, color="blue", lw=3.5)
        #_plt.xticks(_N.arange(0, t1-t0, 1000), _N.arange(t0, t1, 1000, dtype=_N.float)/1000)
        #_plt.xticks([0, t1-t0], [t0, t1])
        sts = mkd.sts
        ists = _N.where((sts > t0) & (sts < t1))[0]
        if mode == 0:  #  don't do this for all tetrode mode - just solid black line
            for i in ists[::thnspks]:
                t = sts[i]
        #        _plt.plot([(t-t0)/skp_disp, (t-t0)/skp_disp], [-3.8, -1.1], color="black")

        #mF.arbitraryAxes(ax, axesVis=[True, True, True, True], xtpos="bottom", ytpos="left")




    #     ipg = 0
    #     ifg = 1
    #     ppg = 20
    #     pg  = 1
    #     evry = 5


    dmp = open("%(df)s/pX_Nm.dmp" % {"df" : outdirN}, "wb")
    # #  0   posterior position
    # #  1   physical position
    # #  2   likelihood
    # #  3   spike times
    # #  4   epochs


    
    if usemaze == _mkd.mz_W:
        t_03  = _N.where((mkd.pos >= 0) & (mkd.pos < 3))[0]
        t_m03 = _N.where((mkd.pos <= 0) & (mkd.pos > -3))[0]
        t_35  = _N.where((mkd.pos >= 3) & (mkd.pos < 5))[0]
        t_m35 = _N.where((mkd.pos <= -3) & (mkd.pos > -5))[0]
        t_56  = _N.where((mkd.pos >= 5) & (mkd.pos < 6))[0]
        t_m56 = _N.where((mkd.pos <= -5) & (mkd.pos > -6))[0]

        linrzd_pos = _N.zeros(intvs[-1], dtype=_N.float16)

        linrzd_pos[t_03]  = mkd.pos[t_03]
        linrzd_pos[t_m03] = -mkd.pos[t_m03]
        linrzd_pos[t_35]  = 3 + 3-mkd.pos[t_35]
        linrzd_pos[t_m35] = 3 + 3+mkd.pos[t_m35]
        linrzd_pos[t_56]  = 6-mkd.pos[t_56]
        linrzd_pos[t_m56] = 6+mkd.pos[t_m56]

        t0 = intvs[lagfit]
        t1 = intvs[EPCHS]
        #_plt.plot(linrzd_pos[t0:]*(17./3))

        #########  
        Nx_m1 = Nx - 1
        
        #pX_lin = _N.zeros((intvs[-1], 17))   # to be saved
        #pX_lin = _N.zeros((intvs[-1], 16))   # to be saved

        m6     = 0
        m5     = Nx_m1 // 12
        m4     = 2*(Nx_m1 // 12)
        m3     = 3*(Nx_m1 // 12)
        m2     = 4*(Nx_m1 // 12)
        m1     = 5*(Nx_m1 // 12)
        zr     = 6*(Nx_m1 // 12)
        p1     = 7*(Nx_m1 // 12)
        p2     = 8*(Nx_m1 // 12)
        p3     = 9*(Nx_m1 // 12)
        p4     = 10*(Nx_m1 // 12)
        p5     = 11*(Nx_m1 // 12)
        p6     = 12*(Nx_m1 // 12)
        pX_Nm_01  = _N.zeros((intvs[EPCHS]-t0, Nx_m1//12+1))
        pX_lin = _N.zeros((intvs[EPCHS], Nx_m1//4+1))   # to be saved
        
        
        #  0->1, 5->6    0->-1, -5-> -6
        #pX_Nm_01  += mkd.pX_Nm[t0:, 30:36]    #0->1
        pX_Nm_01  += mkd.pX_Nm[t0:t1, zr:p1+1]    #0->1
        #pX_Nm_01  += mkd.pX_Nm[t0:t1, 30:24:-1] #-1->0
        pX_Nm_01  += mkd.pX_Nm[t0:t1, zr:m1-1:-1] #-1->0        
        #pX_Nm_01  += mkd.pX_Nm[t0:, 0:6]      #-5->-6
        pX_Nm_01  += mkd.pX_Nm[t0:t1, m6:m5+1]      #-5->-6
        #pX_Nm_01  += mkd.pX_Nm[t0:, 60:54:-1] #6->5
        pX_Nm_01  += mkd.pX_Nm[t0:t1, p6:p5-1:-1] #6->5

        # pX_Nm_13L = _N.zeros((intvs[-1] - t0, 11))
        # pX_Nm_13L  += mkd.pX_Nm[t0:t1, 35:46]
        # pX_Nm_13L  += mkd.pX_Nm[t0:t1, 55:44:-1] 

        # #  1->3 
        # pX_Nm_13R = _N.zeros((intvs[-1] - t0, 11))
        # pX_Nm_13R  += mkd.pX_Nm[t0:t1, 5:16]    # 
        # pX_Nm_13R  += mkd.pX_Nm[t0:t1, 25:14:-1]


        
        pX_Nm_13L = _N.zeros((intvs[EPCHS] - t0, (Nx_m1//12)*2))
        pX_Nm_13L  += mkd.pX_Nm[t0:t1, p1+1:p3+1]
        pX_Nm_13L  += mkd.pX_Nm[t0:t1, p5:p3:-1]  # don't want to overcount 45

        #  1->3 
        pX_Nm_13R = _N.zeros((intvs[EPCHS] - t0, (Nx_m1//12)*2))
        pX_Nm_13R  += mkd.pX_Nm[t0:t1, m5+1:m3+1]    # 
        pX_Nm_13R  += mkd.pX_Nm[t0:t1, m1:m3:-1]
        

        pX_lin[t0:t1, 0:(Nx_m1//12)+1] = pX_Nm_01
        pX_lin[t0:t1, (Nx_m1//12)+1:] += pX_Nm_13R
        pX_lin[t0:t1, (Nx_m1//12)+1:] += pX_Nm_13L

        start_t = _N.where(_N.sum(mkd.Lklhd[0, mkd.sts], axis=1) == 0)[0][-1]
        mkd.Lklhd[0, mkd.sts[0:start_t+1]] = 1e-100
        _pkl.dump([saveBin,
                   _N.asfarray(mkd.pX_Nm[::saveBin], dtype=_N.float16),
                   _N.asfarray(mkd.pos[::saveBin], dtype=_N.float16),
                   _N.asfarray(_N.log(mkd.Lklhd[0, mkd.sts]), _N.float16),
                   mkd.sts,
                   intvs,
                   silenceLklhds,
                   _N.asfarray(linrzd_pos[::saveBin], dtype=_N.float16),
                   _N.asfarray(pX_lin[::saveBin], dtype=_N.float16)], dmp, -1)
    else:
        start_t = _N.where(_N.sum(mkd.Lklhd[0, mkd.sts], axis=1) == 0)[0][-1]
        mkd.Lklhd[0, mkd.sts[0:start_t+1]] = 1e-100
        _pkl.dump([saveBin, _N.asfarray(mkd.pX_Nm[::saveBin], dtype=_N.float16), _N.asfarray(mkd.pos[::saveBin], dtype=_N.float16), _N.asfarray(_N.log(mkd.Lklhd[0, mkd.sts]), _N.float16), mkd.sts, intvs, silenceLklhds], dmp, -1)

    dmp.close()

    _plt.ion()    


