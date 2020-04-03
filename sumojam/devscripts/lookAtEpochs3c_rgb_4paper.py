import mvn
from jifimogs.tools.JIFIdirs import resFN, datFN
import pickle as _pkl
import jifimogs.tools.plottools as mF
from matplotlib.ticker import FormatStrFormatter
import numpy as _N
import matplotlib.pyplot as _plt
import jifimogs.tools.utils as _util
import matplotlib.ticker as ticker

tksz = 21
epcs = 15
K    = 2
KGT  = 4      #  actual K
N    = 80     #  resolution of imshow.  interpolation="None" reduces file size drastically.
onepage=False
saturate = False
useEstMode = True
satLvl = 0.4
useGT    = False
satpct= 0.85
smpITER      = 2
thin    = 1
#  mk_hypPstMd   is epc x M x K

#tet = 106
tet = 1
datfn = "bc2_mdec0101"
#datfn = "frank_mdec0202"
epchfn= "15_2"
label = "1"
disp_pct = 0.99

fnSMP = ""

#_plt.ioff()

ytick_pos = "left"
imgfmt = "eps"
dfn = "%(df)s-%(ef)s" % {"df" : datfn, "ef" : epchfn}
if label is not None:
    dfn += "-%s" % label
if tet is not None:
    dfn += "/%d" % tet

# uses = [[0, 1], [0, 2], [0, 3], [0, 4],
#                 [1, 2], [1, 3], [1, 4],
#                         [2, 3], [2, 4],
#                                 [3, 4]]

# """
# uses = [[1, 4]]
# """

spis = [1, 2, 3]
if datfn == "bond_mdec0302":
    if tet == 112:
        uses = [[0, 1], [0, 2], [1, 2]]

        disp_ticks = _N.array([[-1, -1, -1, -1],
                               [0, 50, 100, 150],   # -15 162
                               [-50, 80, 210, 340]]) # -69 353
    elif tet == 114:
        uses = [[0, 2], [0, 4], [2, 4]]

        disp_ticks = _N.array([[-1, -1, -1, -1],
                               [-1, -1, -1, -1],
#                               [-50, 70, 190, 310],   # -15 162
                               #[-20, 85, 190, 295],   # -15 162
                               [0, 80, 160, 240],   # -15 162
                               [-1, -1, -1, -1],
                               [-50, 150, 350, 550]]) # -69 353
                                   
elif datfn == "frank_mdec0202":
    uses = [[0, 1], [0, 3], [1, 3]]            

    disp_ticks = _N.array([[-1, -1, -1, -1],  # dummy
                           [20, 45, 70, 95], # 19.975 90.835
                           [-1, -1, -1, -1],  # dummy
                           [35, 60, 85, 110]]) # 33.51 98.38

elif datfn == "bc2_mdec0101":
    uses = [[0, 1], [0, 3], [1, 3]]            

    disp_ticks = _N.array([[-1, -1, -1, -1],  # dummy
                           [ 55, 94, 133, 172], # 19.975 90.835
                           [-1, -1, -1, -1],  # dummy
                           [55, 105, 155, 205]]) # 33.51 98.38
            
"""
uses = [[1, 4]]
"""

showonly_this  = None
#showonly_this = _N.array([9])
#rmclstrs = _N.array([4, 16, 29, 30])
#rmclstrs = _N.array([0,1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28, 29, 30,31,32])
#rmclstrs = _N.array([0, 1, 2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])
#rmclstrs = _N.array([0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32])
rmclstrs  = None

print(N)

p_rgb   = _N.zeros((epcs, len(uses), N, N, 3))

if tet is not None:
    dat   = _N.loadtxt(datFN("%(dfn)s_%(tet)d.dat" % {"dfn" : datfn, "tet" : tet}))
else:
    dat   = _N.loadtxt(datFN("%(dfn)s.dat" % {"dfn" : datfn,}))

if useGT:
    if tet is None:
        lm =  _util.depickle(datFN("%s_prms.pkl" % datfn))
    else:
        lm =  _util.depickle(datFN("%(dfn)s_%(tet)d_prms.pkl" % {"dfn" : datfn, "tet" : tet}))
    intvs01         = _N.loadtxt(datFN("itv%s.dat" % epchfn))
    intvs = _N.array(intvs01*dat.shape[0], dtype=_N.int)
    GTsmpIntv = lm["intv"]

Ms   = _N.empty(epcs, dtype=_N.int)

for epc in range(epcs):
    if not useGT:
        lm =  _util.depickle("%(d)s/posteriors_%(e)d.dmp" % {"d" : dfn, "e" : epc})
        intvs        = lm["intvs"]

    t0   = intvs[epc]
    t1   = intvs[epc+1]

    #sts = _N.where(dat[t0:t1, 1] == 1)[0]   # use all of data

    sts = _N.where(dat[intvs[0]:intvs[epcs], 1] == 1)[0]+intvs[0] #  all epochs - so we can easily compare plots between epoch because they use same axis limits
    sd  = _N.sort(dat[sts, 2:], axis=0)    #  first index is tetrode

    #sds = _N.std(dat[sts, 2:], axis=0)
    #mns = _N.median(dat[sts, 2:], axis=0)

    Nsp  = len(sts)
    #amps= sds*4  #  4 standard devs
    #mins= mns - amps;   maxs = mns + amps
    amps= sd[int(Nsp*disp_pct)] - sd[int(Nsp*(1-disp_pct))]

    mins = sd[int(Nsp*(1-disp_pct))] - 0.5*amps
    maxs = sd[int(Nsp*disp_pct)] + 0.5*amps    
#     mins= _N.min(sd, axis=0) - 0.2*amps;     maxs= _N.max(sd, axis=0) + 0.2*amps

    Wdth= sd[-1] - sd[0]

    #amps = maxs - mins
    #dA   = 0.05*amps
    #x_L  = _N.array([-6, mins[0]-dA[0], mins[1]-dA[1], mins[2]-dA[2], mins[3]-dA[3]])
    #x_H  = _N.array([ 6, maxs[0]+dA[0], maxs[1]+dA[1], maxs[2]+dA[2], maxs[3]+dA[3]])
    #if K == 2:
    #x_L  = _N.array([-6, mins[0], mins[1], ])
    #    x_H  = _N.array([ 6, maxs[0], maxs[1], ])
    #if K == 4:
    x_L  = _N.array([-6, mins[0], mins[1], mins[2], mins[3]])
    x_H  = _N.array([ 6, maxs[0], maxs[1], maxs[2], maxs[3]])


    
    if (not useGT) and useEstMode:
        M            = lm["M"]
        if "freeClstr" in lm:
            inuse_clstr = _N.where(lm["freeClstr"] == False)[0]
        else:
            inuse_clstr = _N.arange(M)
        M            = len(inuse_clstr)
        Mwowonz      = M

        Ms[epc]      = M
        uss          = lm["mk_prmPstMd"][0][inuse_clstr]
        Sgs          = lm["mk_prmPstMd"][1][inuse_clstr]
        l0s          = lm["sp_prmPstMd"][::3][inuse_clstr]

            
        #ix = _N.where(l0s < 0)
        #l0s[ix] = 0.001

        if showonly_this is not None:
            rmclstrs = _N.setdiff1d(_N.arange(M), showonly_this)
        if rmclstrs is not None:
            l0s[rmclstrs] = 0

        fss          = lm["sp_prmPstMd"][1::3][inuse_clstr]
        q2s          = lm["sp_prmPstMd"][2::3][inuse_clstr]
        # print((l0s/_N.sqrt(q2s)))
        # print(_N.linalg.det(Sgs))
        # print("----------------")
        

        #nz_pth       = lm["nz_pth"]
    elif useGT:
        t0 =    intvs[epc] // GTsmpIntv
        t1 =    intvs[epc+1] // GTsmpIntv

        kms = lm["km"]    
        M  = max([item for sublist in kms for item in sublist]) + 1

        uss = _N.empty((M, KGT))
        Sgs = _N.empty((M, KGT, KGT))
        l0s = _N.empty(M)
        fss = _N.empty(M)
        q2s = _N.empty(M)
        im  = -1
        for inrn in range(len(kms)):
            for ii in range(len(kms[inrn])):
                im += 1
                arr = _N.array(lm["u"][inrn, t0:t1])
                uss[im]          = _N.mean(arr, axis=0)
                Sgs[im]          = lm["covs"][inrn]
                l0s[im]          = _N.mean(lm["l0"][im, t0:t1], axis=0)
                fss[im]          = _N.mean(lm["f"][im, t0:t1], axis=0)
                q2s[im]          = _N.mean(lm["sq2"][im, t0:t1], axis=0)
        ix = _N.where(l0s < 0)
        l0s[ix] = 0.001

    else:   # use last sample
        smps = lm["smp_sp_prms"]
        mmps = lm["smp_mk_prms"]

        fnSMP        = "-%s" % smpITER
        uss          = mmps[0][:, smpITER].T#reshape(1, M, K)
        Sgs          = mmps[1][:, :, smpITER].T#.reshape(1, M, K, K)
        l0s          = smps[0, smpITER]#.reshape((1, M))
        fss          = smps[1, smpITER]#.reshape((1, M))
        q2s          = smps[2, smpITER]#.reshape((1, M))
        M            = smps.shape[2]#lm["sp_prmPstMd"].shape[1]/3

    ii    = -1

    for use in uses:
        ii += 1
        labX= "pos" if (use[0] == 0) else "mk tet%d" % use[0]
        labY= "pos" if (use[1] == 0) else "mk tet%d" % use[1]

        x1   = _N.linspace(x_L[use[0]], x_H[use[0]], N)
        x2   = _N.linspace(x_L[use[1]], x_H[use[1]], N)
        amps = x_H - x_L

        xy  = [x1, x2]

        grid = _N.array(_N.meshgrid(*xy, indexing="ij"))
        Ndat = dat.shape[0]

        Sg  = _N.zeros((2, 2))
        mns = _N.empty(2)
        mnsr = mns.reshape((2, 1, 1))

        for m in range(M):
            for ch in range(2):
                if use[ch] == 0:
                    Sg[ch, ch]      =   q2s[m]
                    mnsr[ch, 0, 0]  =   fss[m]
                else:
                    mkd = use[ch]-1
                    Sg[ch, ch]      =   Sgs[m, mkd, mkd]
                    mnsr[ch, 0, 0] =   uss[m, mkd]

            if (use[0] > 0) and (use[1] > 0):
                mkd1 = use[0] - 1;          mkd2 = use[1] - 1
                Sg[0, 1]   =   Sgs[m, mkd1, mkd2]
                Sg[1, 0]   =   Sgs[m, mkd2, mkd1]
            iSg = _N.linalg.inv(Sg)

            xyg = (grid-mnsr).T


            if _N.linalg.det(Sg) > 0:
                p_rgb[epc, ii, :, :, 0] +=  _N.sqrt((l0s[m] / _N.sqrt(2*_N.pi*_N.linalg.det(Sg)) * _N.exp(-0.5*_N.einsum("ijk,kl,ijl->ij", xyg, iSg, xyg))))
            else:
                print("Sg of cluster %d is not positive definite" % m)
                print("use[0]=%(0)d   use[1]=%(1)d" % {"0" : use[0], "1" : use[1]})
                print(Sg)

        if saturate and satpct < 1.:
            # fp = p[epc, ii].flatten()
            # thr = _N.sort(fp)[int(len(fp)*satpct)]
            # inds = _N.where(p[epc, ii] > thr)
            #p[epc, ii, inds[0], inds[1]] = thr
            #p_rgb = _N.log(1 + p_rgb)
            p_rgb[:, :, :, :, 0]  = _N.sqrt(p_rgb[:, :, :, :, 0])
#amp     = _N.max(_N.abs(p))    #  maximum amplitude

p_rgb[:, :, :, :, 1] = p_rgb[:, :, :, :, 0]
p_rgb[:, :, :, :, 2] = p_rgb[:, :, :, :, 0]
#p_rgb[:, :, :, :, 0] /= _N.max(p_rgb[:, :, :, :, 0])
for iu in range(len(uses)):
    p_rgb[:, iu, :, :] /= _N.max(p_rgb[:, iu, :, :, 0])


#############################  
##########################
for epc in range(epcs):
    t0   = intvs[epc]
    t1   = intvs[epc+1]

    sts = _N.where(dat[t0:t1, 1] == 1)[0] + t0
    mkpos = _N.empty((len(sts), 2))

    for fign in range(1, 3):
        ii = -1
        #fig  = _plt.figure(num=fign, figsize=(12.2, 12))
        #fig  = _plt.figure(num=fign, figsize=(13.2, 12.7))  # if adding labels
        fig  = _plt.figure(num=fign, figsize=(4.5, 13.2))  # if adding labels

        for use in uses:
            ii += 1
            x1   = _N.linspace(x_L[use[0]], x_H[use[0]], N)
            x2   = _N.linspace(x_L[use[1]], x_H[use[1]], N)


            ch1  = 0 if use[0] == 0 else use[0] + 1
            ch2  = 0 if use[1] == 0 else use[1] + 1
            mkpos[:, 0] = (dat[sts, ch1] - x_L[use[0]]) * (N/float(amps[use[0]]))
            mkpos[:, 1] = (dat[sts, ch2] - x_L[use[1]]) * (N/float(amps[use[1]]))
            ax = fig.add_subplot(3, 1, spis[ii])
            #_plt.title("epoch %d" % epc, fontsize=tksz)
            if fign == 1:
                scps = 5 if K == 4 else 11
                ax.scatter(mkpos[::thin, 0], mkpos[::thin, 1], s=scps, color="orange")
            #ax.imshow(p[epc, ii], origin="lower", cmap=_plt.get_cmap("Blues"), vmin=0, vmax=_N.max(p[:, ii]), interpolation="none")

            ax.imshow(1-p_rgb[epc, ii], origin="lower", interpolation="none")

            #if fign == 2:
            #ax.contour(_N.arange(N), _N.arange(N), p[0, ii], levels=_N.array([_N.mean(p[:, ii])*satLvl]), colors=["#333333"])
            #_plt.xticks(, _N.linspace(x_L[use[0]], x_H[use[0]], 3), fontsize=tksz)
            #_plt.yticks([0, N/2., N], _N.linspace(x_L[use[1]], x_H[use[1]], 3), fontsize=tksz)

#plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e')) 
            #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))


            if ytick_pos == "left":
                mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos=ytick_pos)
            elif ytick_pos == "right":
                mF.arbitraryAxes(ax, axesVis=[False, True, True, False], xtpos="bottom", ytpos=ytick_pos)

            #mF.setTicksAndLims(xlabel=labX, ylabel=labY, xticks=None, yticks=None, xticksD=None, yticksD=None, xlim=None, ylim=None, tickFS=26, labelFS=28)

            # xlab = None
            # ylab = None
            # if ((spis[ii] % K) == 1):
            #     ylab = "elctrd"
            # if (spis[ii] == K*(K-1)+1):
            #     print "i am here"
            #     xlab = "position"
            # elif (spis[ii] > K*(K-1)+1):
            #     xlab = "elctrd"

            if use[0] == 0:
                #mF.setTicksAndLims(xlabel=None, ylabel=None, xticks=[0, 0.25*N, 0.5*N, 0.75*N, N], yticks=[0, N/2., N], xticksD=["H", "L", "H", "R", "H"], yticksD=_N.round(_N.linspace(x_L[use[1]], x_H[use[1]], 3), 2), xlim=None, ylim=None, tickFS=tksz, labelFS=(tksz+2))
                yticks_disp = disp_ticks[use[1]]
                y_uV_H = x_H[use[1]]
                y_uV_L = x_L[use[1]]
                yticks = (N / (y_uV_H - y_uV_L)) * (yticks_disp - y_uV_L)
                print("yticks!!!!    %d", use[0])
                print(yticks)
                
                #mF.setTicksAndLims(xlabel="position", ylabel=("mark ch %d" % use[1]), xticks=[0, 0.25*N, 0.5*N, 0.75*N, N], xticksD=["H", "L", "H", "R", "H"], yticks=yticks, yticksD=yticks_disp, xlim=None, ylim=None, tickFS=tksz, labelFS=(tksz+2))
                
                if (yticks[0] < 0) or (yticks[-1] > N):
                    print("!!!!!!!!!!  ticks not right, images might be wrong dimensions")
                    print(yticks)
                mF.setTicksAndLims(xlabel="position", ylabel=("mark ch %d $(\\mu V)$" % use[1]), xticks=[0, 0.25*N, 0.5*N, 0.75*N, N], xticksD=["H", "L", "H", "R", "H"], yticks=yticks, yticksD=yticks_disp, xlim=None, ylim=None, tickFS=tksz, labelFS=(tksz+2))

            else:
                #mF.setTicksAndLims(xlabel=None, ylabel=None, xticks=[0, N/2., N], yticks=[0, N/2., N], xticksD=_N.round(_N.linspace(x_L[use[0]], x_H[use[0]], 3), 2), yticksD=_N.round(_N.linspace(x_L[use[1]], x_H[use[1]], 3), 2), xlim=None, ylim=None, tickFS=tksz, labelFS=(tksz+2))
                #  0 -->  x_L
                #  N -->  x_H
                xticks_disp = disp_ticks[use[0]]
                yticks_disp = disp_ticks[use[1]]
                x_uV_H = x_H[use[0]]
                y_uV_H = x_H[use[1]]
                x_uV_L = x_L[use[0]]
                y_uV_L = x_L[use[1]]                
                
                xticks = (N / (x_uV_H - x_uV_L)) * (xticks_disp - x_uV_L)
                yticks = (N / (y_uV_H - y_uV_L)) * (yticks_disp - y_uV_L)

                mF.setTicksAndLims(xlabel=("mark ch %d" % use[0]), xticks=xticks, xticksD=xticks_disp, yticks=yticks, yticksD=yticks_disp, ylabel=("mark ch %d" % use[1]), xlim=[0, N], ylim=[0, N], tickFS=tksz, labelFS=(tksz+2))

                #mF.setTicksAndLims(xlabel=None, xticks=xticks, xticksD=xticks_disp, yticks=yticks, yticksD=yticks_disp, ylabel=None, xlim=[0, N], ylim=[0, N], tickFS=tksz, labelFS=(tksz+2))                
                

                #ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
                #ax.xaxis.set_major_locator(ticker.MultipleLocator(20))      
                #mF.setTicksAndLims(xlabel=None, ylabel=None, tickFS=tksz, labelFS=(tksz+2))


        fn_of = "GTonlyfit" if useGT else "onlyfit"
        fn_sf = "GTscatterfit" if useGT else "scatterfit"
        if showonly_this is not None:
            fn_of = "%(fn)s_cntrb_%(m)d" % {"fn" : fn_of, "m" : showonly_this[0]}
            fn_sf = "%(fn)s_cntrb_%(m)d" % {"fn" : fn_sf, "m" : showonly_this[0]
}


        # for fign in range(1, 3):
        #     fig  = _plt.figure(num=fign)
        if ytick_pos == "left":
            fig.subplots_adjust(left=0.24, right=0.96, bottom=0.06, top=0.95, wspace=0.5)
        else:
            fig.subplots_adjust(left=0.04, right=0.76, bottom=0.06, top=0.95, wspace=0.5)

        if fign == 1:
            _plt.savefig("%(eb)s/%(fn)s_%(ep)d%(sm)s_%(lr)s.%(fmt)s" % {"eb" : dfn, "ep" : epc, "sm" : fnSMP, "fn" : fn_sf, "fmt" : imgfmt, "lr" : ytick_pos}, transparent=True)
        else:
            _plt.savefig("%(eb)s/%(fn)s_%(ep)d%(sm)s_%(lr)s.%(fmt)s" % {"eb" : dfn, "ep" : epc, "sm" : fnSMP, "fn" : fn_of, "fmt" : imgfmt, "lr" : ytick_pos}, transparent=True)
        _plt.close()


    # #  Place prior for freeClstr near new non-hash spikes that are far 
    # #  from known clusters that are not hash clusters 

    # #  show me the non-hash new spikes in this epoch that are near known 
    # #  non-hash clusters

    #ch  = 0 if use[1] == 0 else use[1] + 1
    mkpos[:, 0]  = dat[sts, 0]
    mkpos[:, 1] = dat[sts, 2]


_plt.ion()

