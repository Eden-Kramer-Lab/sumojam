#  position dependent firing rate
#  discontinuous change of place field params
import os
import utilities as _U
from jifimogs.tools.utils import createSmoothedPath, createSmoothedPathK
import numpy as _N
import matplotlib.pyplot as _plt
import pickle
import time as _tm

UNIF  = 0
NUNIF = 1
GIVEN = 2

APPEAR=0
DISAPPEAR=1
NEITHER=-1
#_N.array([[0, 0.02], [0.1, 0.03], [0.2, 0.04], [0.5, 0.05], [0.7, 0.06]]
def chpts(npts, yL, yH, t0=0, ad=NEITHER):
    #  [tn, dy]
    cpts = _N.empty((npts, 2))
    ts   = _N.random.rand(npts)
    ts[0] = 0
    for i in range(1, npts):
        ts[i] += ts[i-1]
    ts /= _N.sum(ts[-1])
    cpts[:, 0] = ts
    F = 0.9999
    x = _N.random.randn()
    cpts[0, 1] = x
    for i in range(1, npts):
        x = F*x + 0.1*_N.random.randn()
        cpts[i, 1] = x
    miny = _N.min(cpts[:, 1])
    cpts[:, 1] -= miny   # min value is now 0
    maxy = _N.max(cpts[:, 1])
    cpts[:, 1] /= maxy   # max value is now 1
    cpts[:, 1] *= (yH-yL)   # max value is now 1
    cpts[:, 1] += yL
    if ad != NEITHER:
        if ad == APPEAR:
            cpts[0:t0, 1] = 0
        else:
            cpts[t0:, 1]  = 0
    return cpts

        
def makeCovs(nNrns, K, wvfmsz):
    Covs = _N.empty((nNrns, K, K))

    for n in range(nNrns):
        for k1 in range(K):
            #Covs[n, k1, k1] = (LoHisMk[n, k1, 1] - LoHisMk[n, k1, 0])*(0.1+0.1*_N.random.rand())
            Covs[n, k1, k1] = (wvfmsz[n, k1, 0] + (wvfmsz[n, k1, 1] - wvfmsz[n, k1, 0])*_N.random.rand())**2
        for k1 in range(K):
            for k2 in range(k1+1, K):
                #Covs[n, k1, k2] = (0.65+0.25*_N.random.rand()) * _N.sqrt(Covs[n, k1, k1]*Covs[n, k2, k2])#(0.5 + 0.3*_N.random.rand()) * _N.sqrt(Covs[n, k1, k1]*Covs[n, k2, k2])
                Covs[n, k1, k2] = (0.84+0.1*_N.random.rand()) * _N.sqrt(Covs[n, k1, k1]*Covs[n, k2, k2])#(0.5 + 0.3*_N.random.rand()) * _N.sqrt(Covs[n, k1, k1]*Covs[n, k2, k2])
                Covs[n, k2, k1] = Covs[n, k1, k2]
    return Covs

def create(Lx, Hx, N, mvPat, RTs, frqmx, Amx, pT, l_sx_chpts, l_l0_chpts, l_ctr_chpts, mk_chpts, Covs, LoHis, km, bckgrdLam=None, script="no info", addShortStops=False, stops=10, stopDur=500, thresh=None, x_mvt=None, nz_mvt=None, spc_dim=1, segs=None):
    """
    km  tells me neuron N gives rise to clusters km[N]  (list)
    bckgrd is background spike rate  (Hz)
    """
    global UNIF, NUNIF
    #####  First check that the number of neurons and PFs all consistent.
    nNrnsSX = len(l_sx_chpts)
    nNrnsL0 = len(l_l0_chpts)
    nNrnsCT = len(l_ctr_chpts)
    nNrnsMK = len(mk_chpts)
    nNrnsMKA= LoHis.shape[0]
    nNrnsMKC= Covs.shape[0]

    if not (nNrnsSX == nNrnsL0 == nNrnsCT == nNrnsMK == nNrnsMKA == nNrnsMKC):
        print("Number of neurons not consistent")
        return None
    nNrns = nNrnsSX

    if not (LoHis.shape[1] == Covs.shape[1] == Covs.shape[2]):
        print("Covariance of LoHis not correct")
        return None
    K = LoHis.shape[1]
    
    PFsPerNrn = _N.zeros(nNrns, dtype=_N.int)

    sx_chpts  = []
    l0_chpts  = []
    ctr_chpts = []
    M         = 0    #  of place fields total.  a neuron may have > 1 PFs.
    nrnNum    = []
    for nrn in range(nNrns):
        #  # of place fields for neuron nrn
        nPFsSX = len(l_sx_chpts[nrn])
        nPFsL0 = len(l_l0_chpts[nrn])
        nPFsCT = len(l_ctr_chpts[nrn])
        sx_chpts.extend(l_sx_chpts[nrn])
        l0_chpts.extend(l_l0_chpts[nrn])
        ctr_chpts.extend(l_ctr_chpts[nrn])

        if not (nPFsSX == nPFsL0 == nPFsCT):
            print("Number of PFs for neuron %d not consistent" % nrn)
            return None
        M += len(l_ctr_chpts[nrn])
        nrnNum += [nrn]*nPFsSX
        PFsPerNrn[nrn] = nPFsSX

    #  M = # of clusters  (in mark + pos space)  
    #  nNrns = # of neurons

    if x_mvt is None:
        ####  build data
        Ns     = _N.empty(RTs, dtype=_N.int)
        if mvPat == NUNIF:
            for rt in range(RTs):
                Ns[rt] = N*((1-pT) + pT*_N.random.rand())
        else:
            Ns[:] = N

        NT     = _N.sum(Ns)     #  total time we have data
        pths    = _N.empty(NT)
    else:
        NT     = x_mvt.shape[0]     #  total time we have data


    plastic = False

    ##########  nonstationary center width
    #  sx_chpts is flattened version of l_sx_chpts, which is a list per neuron
    #  of place field change points.  neur#1 has 1 pf, neur#2 has 5 pfs = 6 pfs
    #  sxt  should be (M x NT)
    sx = _N.empty((M, NT)) if spc_dim == 1 else _N.empty((M, NT, 2, 2))
    isx = _N.empty((M, NT)) if spc_dim == 1 else _N.empty((M, NT, 2, 2))

    for m in range(M):  # sxts time scale
        if spc_dim == 1:
            sx[m] = createSmoothedPath(sx_chpts[m], NT)**2  #  sx_chpts[m] is an ndaray.  sx_chpts is a list
            if len(sx_chpts[m]) > 1:  plastic = True
            isx[m] = 1./sx[m]
        else:
            sx[m, :, 0, 0] = createSmoothedPath(sx_chpts[m][:, _N.array([0, 1])], NT)**2
            sx[m, :, 1, 0] = createSmoothedPath(sx_chpts[m][:, _N.array([0, 2])], NT)**2
            sx[m, :, 0, 1] = sx[m, :, 1, 0]
            sx[m, :, 1, 1] = createSmoothedPath(sx_chpts[m][:, _N.array([0, 3])], NT)**2
            isx[m] = _N.linalg.inv(sx[m])

    ##########  nonstationary center height l0
    #  f is NT x M
    l0   = _N.empty((M, NT))
    for m in range(M):
        l0[m] = createSmoothedPath(l0_chpts[m], NT)
        if len(l0_chpts[m]) > 1:  plastic = True

    f     = l0/_N.sqrt(2*_N.pi*sx) if spc_dim == 1 else l0/_N.sqrt((2*_N.pi)*(2*_N.pi)*_N.linalg.det(sx))


    ##########  nonstationary center width
    #  sx_chpts is flattened version of l_sx_chpts, which is a list per neuron
    #  of place field change points.  neur#1 has 1 pf, neur#2 has 5 pfs = 6 pfs
    #  sxt  should be (M x NT)
    ctr = _N.empty((M, NT)) if spc_dim == 1 else _N.empty((M, NT, 2))

    for m in range(M):  # sxts time scale
        if spc_dim == 1:
            ctr[m] = createSmoothedPath(ctr_chpts[m], NT)  #  sx_chpts[m] is an ndaray.  sx_chpts is a list
            if len(ctr_chpts[m]) > 1:  plastic = True
        else:
            ctr[m, :, 0] = createSmoothedPath(ctr_chpts[m][:, _N.array([0, 1])], NT)
            ctr[m, :, 1] = createSmoothedPath(ctr_chpts[m][:, _N.array([0, 2])], NT)

    if K > 0:
        ##########  nonstationary marks
        mk_MU  = _N.empty((nNrns, NT, K))
        print("-------  ")

        for n in range(nNrns):
            mk_MU[n] = createSmoothedPathK(mk_chpts[n], NT, K, LoHis[n])

            if len(mk_chpts[n]) > 1:  plastic = True

    if x_mvt is None:
        if mvPat == NUNIF:
            now = 0
            for rt in range(RTs):
                N = Ns[rt]    #  each traverse slightly different duration
                rp  = _N.random.rand(N//100)
                x     = _N.linspace(Lx, Hx, N)
                xp     = _N.linspace(Lx, Hx, N//100)

                r   = _N.interp(x, xp, rp)       #  creates a velocity vector
                #  create movement without regard for place field
                frqmxR = _N.abs(frqmx*(1+0.25*_N.random.randn()))
                _N.linspace(0, 1, N, endpoint=False)
                rscld_t = _N.random.rand(N)   #  rscld_t
                rscld_t /= (_N.max(rscld_t)*1.01)
                rscld_t.sort()
                phi0 = _N.random.rand()*2*_N.pi

                r += _N.exp(Amx*_N.sin(2*_N.pi*rscld_t*frqmxR + phi0))
                pth = _N.zeros(N+1)
                for n in range(1, N+1):
                    pth[n] = pth[n-1] + r[n-1]

                pth   /= (pth[-1] - pth[0])
                pth   *= (Hx-Lx)
                pth   += Lx

                pths[now:now+N]     = pth[0:N]
                now += N
        else:  # x_mvt is not None 
            now = 0
            x = _N.linspace(Lx, Hx, N)
            for rt in range(RTs):
                N = Ns[rt]
                pths[now:now+N]     = x
                now += N

        if addShortStops:
            for ist in range(stops):
                done   = False
                while not done:
                    t0 = int(_N.random.rand()*NT)
                    t1 = t0 + int(stopDur*(1+0.1*_N.random.randn()))
                    if _N.abs(_N.max(_N.diff(pths[t0:t1]))) < 0.05*(Hx-Lx):
                        done = True   #  not crossing origin

                pths[t0:t1] = _N.mean(pths[t0:t1])
    else:
        pths = x_mvt

    ###  now calculate firing rates
    dt   = 0.001
    fdt  = f*dt
    #  change place field location
    if spc_dim == 1:
        Lam   = f*dt*_N.exp(-0.5*(pths-ctr)**2 / sx)
    else:   #  spc_dim == 2
        Lam   = _N.empty((M, NT))
        for m in range(M):
            dif = pths - ctr[m]
            Lam[m] = f[m]*dt*_N.exp(-0.5*_N.einsum("ni,nij,nj->n", dif, isx[m], dif))
    _N.savetxt("lam", Lam.T)
    _N.savetxt("pths", pths)

    rnds = _N.random.rand(M, NT)

    if spc_dim == 1:
        dat = _N.zeros((NT, 2 + K+1))
        dat[:, 0] = pths
        if segs is not None:
            dat[:, 2+K] = segs
        else:
            dat[:, 2+K] = 1
    else:
        dat = _N.zeros((NT, 3 + K+2))
        if nz_mvt is not None:
            dat[:, 0:2] = nz_mvt
        else:
            dat[:, 0:2] = pths
        dat[:, 7:9] = segs

    spkind = 1 if spc_dim == 1 else 2
    mrk_frm= 2 if spc_dim == 1 else 3
    for m in range(M):
        sts  = _N.where(rnds[m] < Lam[m])[0]      #  spikes from this neuron
        print("neuron %(m)d   %(s)d spks" % {"m" : m, "s" : len(sts)})
        alrdyXst = _N.where(dat[:, spkind] == 1)[0]  #  places where synchronous
        ndToMove = _N.intersect1d(alrdyXst, sts)
        dntMove  = _N.setdiff1d(sts, ndToMove)    #  empty slots
        nonsynch = _N.empty(len(sts))          
        nonsynch[0:len(dntMove)]    = dntMove
        print(len(ndToMove))

        iStart   = len(dntMove)                   #  in nonsynch

        
        # for iOcpd in ndToMove:  # occupied
        #     bDone = False
        #     while not bDone:
        #         iOcpd += 1
        #         if datNghbr[iOcpd, 1] == 0:
        #             datNghbr[iOcpd, 1] = 1
        #             nonsynch[iStart]   = iOcpd
        #             iStart += 1
        #             bDone = True

        snonsynch = _N.sort(nonsynch)
        dat[sts, spkind] = 1

        nrn = nrnNum[m]

        if K > 0:
            for t in range(len(sts)):
                obsMrk = _N.random.multivariate_normal(mk_MU[nrn, sts[t]], Covs[nrn], size=1)
                dat[sts[t], mrk_frm:mrk_frm+K] = obsMrk

        #  now noise spikes
        if bckgrdLam is not None:
            nzsts  = _N.where(rnds[m] < (bckgrdLam*dt)/float(M))[0]
            dat[nzsts, spkind] = 1
            nrn = nrnNum[m]
            if K > 0:
                for t in range(len(nzsts)):
                    dat[nzsts[t], mrk_frm:mrk_frm+K] = _N.random.multivariate_normal(mk_MU[nrn, nzsts[t]], Covs[nrn], size=1)

    if thresh is not None:
        sts = _N.where(dat[:, spkind] == 1)[0]
        nID, nC = _N.where(dat[sts, mrk_frm:mrk_frm+K] < thresh)

        swtchs  = _N.zeros((len(sts), K))
        swtchs[nID, nC] = 1    #  for all cells, all components below hash == 1

        swtchsK = _N.sum(swtchs, axis=1)
        
        blw_thrsh_all_chs = _N.where(swtchsK == K)[0]
        abv_thr = _N.setdiff1d(_N.arange(len(sts)), blw_thrsh_all_chs)
        print("below thresh in all channels  %(1)d / %(2)d" % {"1" : len(blw_thrsh_all_chs), "2" : len(sts)})
        dat[sts[blw_thrsh_all_chs], spkind] = 0

    bFnd  = False

    ##  us un   uniform sampling of space, stationary or non-stationary place field
    ##  ns nn   non-uni sampling of space, stationary or non-stationary place field
    ##  bs bb   biased and non-uni sampling of space

    bfn     = "" if (M == 1) else ("%d" % M)

    if mvPat == UNIF:
        bfn += "u"
    else:
        bfn += "b" if (Amx > 0) else "n"

    bfn += "n" if plastic else "s"

    iInd = 0
    while not bFnd:
        iInd += 1
        dd   = os.getenv("__JIFIDataDir__")
        fn = "%(dd)s/%(bfn)s%(iI)d.dat" % {"bfn" : bfn, "iI" : iInd, "dd" : dd}
        fnocc="%(dd)s/%(bfn)s%(iI)docc.png" % {"bfn" : bfn, "iI" : iInd, "dd" : dd}
        fnprm = "%(dd)s/%(bfn)s%(iI)d_prms.pkl" % {"bfn" : bfn, "iI" : iInd, "dd" : dd}

        if not os.access(fn, os.F_OK):  # file exists
            bFnd = True

    smk = " %.2f" * K

    if spc_dim == 1:
        smk += " %d"
        _U.savetxtWCom("%s" % fn, dat, fmt=("%.4f %d" + smk), delimiter=" ", com="#  script=%s.py" % script)
    else:
        smk += " %d %d"
        _U.savetxtWCom("%s" % fn, dat, fmt=("%.2f %.2f %d" + smk), delimiter=" ", com="#  script=%s.py" % script)

    pcklme = {}

    pcklme["l0"]  = l0[:, ::100]
    pcklme["u"]   = mk_MU[:, ::100]
    pcklme["covs"]= Covs
    pcklme["intv"]= 100
    pcklme["km"]  = km

    pcklme["f"]   = ctr[:, ::100]
    pcklme["sq2"] = sx[:, ::100]

    dmp = open(fnprm, "wb")
    pickle.dump(pcklme, dmp, -1)
    dmp.close()

    print("created %s" % fn)

    # fig = _plt.figure()
    # _plt.hist(dat[:, 0], bins=_N.linspace(Lx, Hx, 101), color="black")
    # _plt.savefig(fnocc)
    # _plt.close()


