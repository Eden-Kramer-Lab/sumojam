import scipy.stats as _ss
import jifimogs.tools.mcmcFigs as mF
import numpy as _N
import matplotlib.pyplot as _plt
from matplotlib.patches import Polygon

tkFS=19
lbFS=21

#dats = [1, 2, 3, 4, 5, 6, 7, 8]
pct=0.99

#
#dats = range(16, 25)

NOCHANGE = False

anims = ["bc20101", "bc30101", "bc40101", "bc50101", "bc60101"]


nhists = 5
xticksEvry = 4
dats = list(range(1, 7))
lastDecodedEp = 14

itv_fn_A    = "2_11"   #  dec ep13  (last epoch)
#    itv_fn_A    = "2_8"   #  dec ep13  (last epoch)
itv_fn_B    = "2_12"   #  dec ep11
itv_fn_C    = "2_13"   #  dec ep9
itv_fn_D    = "2_14"   #  dec ep7
itv_fn_E    = "2_15"   #  dec ep5
itv_fn      = "15_2"
label       = "1"
label_GT       = "1"

r2as         = _N.zeros((len(anims), lastDecodedEp))
r2es         = _N.zeros((len(anims), lastDecodedEp))
c95as        = _N.zeros((len(anims), lastDecodedEp))
c95es        = _N.zeros((len(anims), lastDecodedEp))
w95as        = _N.zeros((len(anims), lastDecodedEp))
w95es        = _N.zeros((len(anims), lastDecodedEp))
r2_1s13as    = _N.zeros(len(anims))
r2_1s13es    = _N.zeros(len(anims))
r2_1s11as    = _N.zeros(len(anims))
r2_1s11es    = _N.zeros(len(anims))
r2_1s9as     = _N.zeros(len(anims))
r2_1s9es     = _N.zeros(len(anims))
r2_1s7as     = _N.zeros(len(anims))
r2_1s7es     = _N.zeros(len(anims))
r2_1s5as     = _N.zeros(len(anims))
r2_1s5es     = _N.zeros(len(anims))
w95_1s13as    = _N.zeros(len(anims))
w95_1s13es    = _N.zeros(len(anims))
w95_1s11as    = _N.zeros(len(anims))
w95_1s11es    = _N.zeros(len(anims))
w95_1s9as     = _N.zeros(len(anims))
w95_1s9es     = _N.zeros(len(anims))
w95_1s7as     = _N.zeros(len(anims))
w95_1s7es     = _N.zeros(len(anims))
w95_1s5as     = _N.zeros(len(anims))
w95_1s5es     = _N.zeros(len(anims))
c95_1s13as    = _N.zeros(len(anims))
c95_1s13es    = _N.zeros(len(anims))
c95_1s11as    = _N.zeros(len(anims))
c95_1s11es    = _N.zeros(len(anims))
c95_1s9as     = _N.zeros(len(anims))
c95_1s9es     = _N.zeros(len(anims))
c95_1s7as     = _N.zeros(len(anims))
c95_1s7es     = _N.zeros(len(anims))
c95_1s5as     = _N.zeros(len(anims))
c95_1s5es     = _N.zeros(len(anims))


alldat_gts   = _N.zeros((len(anims), len(dats), lastDecodedEp, 3))
alldat_mogAs  = _N.zeros((len(anims), len(dats), 3))
alldat_mogBs  = _N.zeros((len(anims), len(dats), 3))
alldat_mogCs  = _N.zeros((len(anims), len(dats), 3))
alldat_mogDs  = _N.zeros((len(anims), len(dats), 3))
alldat_mogEs  = _N.zeros((len(anims), len(dats), 3))


ianim = -1
for anim in anims:
    ianim += 1
    #
    alldat_mog  = _N.zeros((len(dats), lastDecodedEp, 3))
    alldat_gt   = _N.zeros((len(dats), lastDecodedEp, 3))
    alldat_mogA  = _N.zeros((len(dats), 3))
    alldat_mogB  = _N.zeros((len(dats), 3))
    alldat_mogC  = _N.zeros((len(dats), 3))
    alldat_mogD  = _N.zeros((len(dats), 3))
    alldat_mogE  = _N.zeros((len(dats), 3))

    plot_clr = "black"
    for idat in range(len(dats)):
        alldat_mog[idat] = _N.loadtxt("dec-%(an)s-%(itv)s-%(lb)s/MOG_1_%(pct).2f_scrs[%(dn)d].txt" % {"dn" : dats[idat], "itv" : itv_fn, "lb" : label, "pct" : pct, "an" : anim})
        alldat_gt[idat] = _N.loadtxt("dec-%(an)s-%(itv)s-%(lb)s/GT_1_%(pct).2f_scrs[%(dn)d].txt" % {"dn" : dats[idat], "itv" : itv_fn, "lb" : label_GT, "pct" : pct, "an" : anim})
        if nhists >= 1:
            alldat_mogA[idat] = _N.loadtxt("dec-%(an)s-%(itv)s-%(lb)s/MOG_1_%(pct).2f_scrs[%(dn)d].txt" % {"dn" : dats[idat], "itv" : itv_fn_A, "lb" : label, "pct" : pct, "an" : anim})
        if nhists >= 2:
            alldat_mogB[idat] = _N.loadtxt("dec-%(an)s-%(itv)s-%(lb)s/MOG_1_%(pct).2f_scrs[%(dn)d].txt" % {"dn" : dats[idat], "itv" : itv_fn_B, "lb" : label, "pct" : pct, "an" : anim})    
        if nhists >= 3:
            alldat_mogC[idat] = _N.loadtxt("dec-%(an)s-%(itv)s-%(lb)s/MOG_1_%(pct).2f_scrs[%(dn)d].txt" % {"dn" : dats[idat], "itv" : itv_fn_C, "lb" : label, "pct" : pct, "an" : anim})
        if nhists >= 4:
            alldat_mogD[idat] = _N.loadtxt("dec-%(an)s-%(itv)s-%(lb)s/MOG_1_%(pct).2f_scrs[%(dn)d].txt" % {"dn" : dats[idat], "itv" : itv_fn_D, "lb" : label, "pct" : pct, "an" : anim})        
        if nhists >= 5:
            alldat_mogE[idat] = _N.loadtxt("dec-%(an)s-%(itv)s-%(lb)s/MOG_1_%(pct).2f_scrs[%(dn)d].txt" % {"dn" : dats[idat], "itv" : itv_fn_E, "lb" : label, "pct" : pct, "an" : anim})        

    MN  = _N.mean(alldat_mog - alldat_gt, axis=0)
    ER  = _N.std(alldat_mog - alldat_gt, axis=0)
    r2as[ianim] = r2a = MN[:, 2]  # avg
    r2es[ianim] = r2e = ER[:, 2]  # avg

    c95as[ianim] = c95a = MN[:, 0]
    c95es[ianim] = c95e = MN[:, 0]

    w95as[ianim] = w95a = MN[:, 1]
    w95es[ianim] = w95e = MN[:, 1]

    if nhists >= 1:
        mn  = _N.mean(alldat_mogA - alldat_gt[:, lastDecodedEp-1], axis=0) 
        std = _N.mean(alldat_mogA - alldat_gt[:, lastDecodedEp-1], axis=0)
        c95_1s13as[ianim] = c95_1s13a = mn[0]
        c95_1s13es[ianim] = c95_1s13e = std[0]
        w95_1s13as[ianim] = w95_1s13a = mn[1]
        w95_1s13es[ianim] = w95_1s13e = std[1]
        r2_1s13as[ianim]  = r2_1s13a  = mn[2]
        r2_1s13es[ianim]  = r2_1s13e  = std[2]
    if nhists >= 2:
        mn  = _N.mean(alldat_mogB - alldat_gt[:, lastDecodedEp-3], axis=0)
        std = _N.mean(alldat_mogB - alldat_gt[:, lastDecodedEp-3], axis=0)
        c95_1s11as[ianim] = c95_1s11a = mn[0]
        c95_1s11es[ianim] = c95_1s11e = std[0]
        w95_1s11as[ianim] = w95_1s11a = mn[1]
        w95_1s11es[ianim] = w95_1s11e = std[1]
        r2_1s11as[ianim]  = r2_1s11a  = mn[2]
        r2_1s11es[ianim]  = r2_1s11e  = std[2]
    if nhists >= 3:
        mn  = _N.mean(alldat_mogC - alldat_gt[:, lastDecodedEp-5], axis=0)
        std = _N.mean(alldat_mogC - alldat_gt[:, lastDecodedEp-5], axis=0)
        c95_1s9as[ianim] = c95_1s9a = mn[0]
        c95_1s9es[ianim] = c95_1s9e = std[0]
        w95_1s9as[ianim] = w95_1s9a = mn[1]
        w95_1s9es[ianim] = w95_1s9e = std[1]
        r2_1s9as[ianim]  = r2_1s9a  = mn[2]
        r2_1s9es[ianim]  = r2_1s9e  = std[2]
    if nhists >= 4:
        mn  = _N.mean(alldat_mogD - alldat_gt[:, lastDecodedEp-7], axis=0)
        std = _N.mean(alldat_mogD - alldat_gt[:, lastDecodedEp-7], axis=0)
        c95_1s7as[ianim] = c95_1s7a = mn[0]
        c95_1s7es[ianim] = c95_1s7e = std[0]
        w95_1s7as[ianim] = w95_1s7a = mn[1]
        w95_1s7es[ianim] = w95_1s7e = std[1]
        r2_1s7as[ianim]  = r2_1s7a  = mn[2]
        r2_1s7es[ianim]  = r2_1s7e  = std[2]
    if nhists >= 5:
        mn  = _N.mean(alldat_mogE - alldat_gt[:, lastDecodedEp-9], axis=0)
        std = _N.mean(alldat_mogE - alldat_gt[:, lastDecodedEp-9], axis=0)
        c95_1s5as[ianim] = c95_1s5a = mn[0]
        c95_1s5es[ianim] = c95_1s5e = std[0]
        w95_1s5as[ianim] = w95_1s5a = mn[1]
        w95_1s5es[ianim] = w95_1s5e = std[1]
        r2_1s5as[ianim]  = r2_1s5a  = mn[2]
        r2_1s5es[ianim]  = r2_1s5e  = std[2]

    ###########  r2, c95, w95
    fig  = _plt.figure(figsize=(6, 8))
    ##################################################  (3, 1, 1)
    ax   = fig.add_subplot(3, 1, 1)
    _plt.errorbar(range(1, lastDecodedEp+1), r2a, yerr=r2e, fmt='o', ls="-", color=plot_clr, lw=2)
    _plt.axhline(y=0, ls=":", color="grey")

    if nhists >= 1:
        _plt.errorbar([lastDecodedEp-0.19, lastDecodedEp-0.19], [r2_1s13a, r2_1s13a], yerr=r2_1s13e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
    if nhists >= 2:
        _plt.errorbar([lastDecodedEp-2-0.19, lastDecodedEp-2-0.19], [r2_1s11a, r2_1s11a], yerr=r2_1s11e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
    if nhists >= 3:
        _plt.errorbar([lastDecodedEp-4-0.19, lastDecodedEp-4-0.19], [r2_1s9a, r2_1s9a], yerr=r2_1s9e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
    if nhists >= 4:
        _plt.errorbar([lastDecodedEp-6-0.19, lastDecodedEp-6-0.19], [r2_1s7a, r2_1s7a], yerr=r2_1s7e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
    if nhists >= 5:
        _plt.errorbar([lastDecodedEp-8-0.19, lastDecodedEp-8-0.19], [r2_1s5a, r2_1s5a], yerr=r2_1s5e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
        _plt.ylim(-0.5, 1.2)
        _plt.yticks([0, 0.5, 1])
        y_lo, y_hi = ax.get_ylim()
        verts = [(5, y_lo), (5, y_hi), (10, y_hi), (10, y_lo)]
        poly = Polygon(verts, facecolor='0.95', edgecolor='0.95')
        ax.add_patch(poly)
    else:
        _plt.ylim(-0.5, 1.2)

    mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
    mF.setTicksAndLims(ylabel="$\Delta$ rMSE", xticks=range(0, lastDecodedEp+1, xticksEvry), yticksD=None, xlim=[1-0.3, lastDecodedEp+0.3], tickFS=tkFS, labelFS=lbFS)

    ##################################################  (3, 1, 2)
    ax = fig.add_subplot(3, 1, 2)

    _plt.axhline(y=0, ls=":", color="grey")
    _plt.errorbar(range(1, lastDecodedEp+1), c95a, yerr=c95e, fmt='o', ls="-", color=plot_clr, lw=2)
    if nhists >= 1:
        _plt.errorbar([lastDecodedEp-0.19, lastDecodedEp-0.19], [c95_1s13a, c95_1s13a], yerr=c95_1s13e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
    if nhists >= 2:
        _plt.errorbar([lastDecodedEp-2-0.19, lastDecodedEp-2-0.19], [c95_1s11a, c95_1s11a], yerr=c95_1s11e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
    if nhists >= 3:
        _plt.errorbar([lastDecodedEp-4-0.19, lastDecodedEp-4-0.19], [c95_1s9a, c95_1s9a], yerr=c95_1s9e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
    if nhists >= 4:
        _plt.errorbar([lastDecodedEp-6-0.19, lastDecodedEp-6-0.19], [c95_1s7a, c95_1s7a], yerr=c95_1s7e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
    if nhists >= 5:
        _plt.errorbar([lastDecodedEp-8-0.19, lastDecodedEp-8-0.19], [c95_1s5a, c95_1s5a], yerr=c95_1s5e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)

        _plt.ylim(-0.15, 0.1)
        _plt.yticks([-0.1, 0., 0.1])    
        y_lo, y_hi = ax.get_ylim()
        verts = [(5, y_lo), (5, y_hi), (10, y_hi), (10, y_lo)]
        poly = Polygon(verts, facecolor='0.95', edgecolor='0.95')
        ax.add_patch(poly)
    else:
        _plt.ylim(-0.3, 0.1)



    mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
    mF.setTicksAndLims(ylabel="$\\Delta$%% TU%dHPD" % int(pct*100), xticks=range(0, lastDecodedEp+1, xticksEvry), yticksD=None, xlim=[1-0.3, lastDecodedEp+0.3], tickFS=tkFS, labelFS=lbFS)

    ##################################################  (3, 1, 3)
    ax = fig.add_subplot(3, 1, 3)
    _plt.axhline(y=0, ls=":", color="grey")

    _plt.errorbar(range(1, lastDecodedEp+1), w95a, yerr=w95e, fmt='o', ls="-", color=plot_clr, lw=2)
    if nhists >= 1:
        _plt.errorbar([lastDecodedEp-0.19, lastDecodedEp-0.19], [w95_1s13a, w95_1s13a], yerr=w95_1s13e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
    if nhists >= 2:
        _plt.errorbar([lastDecodedEp-2-0.19, lastDecodedEp-2-0.19], [w95_1s11a, w95_1s11a], yerr=w95_1s11e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
    if nhists >= 3:
        _plt.errorbar([lastDecodedEp-4-0.19, lastDecodedEp-4-0.19], [w95_1s9a, w95_1s9a], yerr=w95_1s9e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
    if nhists >= 4:
        _plt.errorbar([lastDecodedEp-6-0.19, lastDecodedEp-6-0.19], [w95_1s7a, w95_1s7a], yerr=w95_1s7e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
    if nhists >= 5:
        _plt.errorbar([lastDecodedEp-8-0.19, lastDecodedEp-8-0.19], [w95_1s5a, w95_1s5a], yerr=w95_1s5e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)

        _plt.ylim(-0.042, 0.06)
        _plt.yticks([-0.04, 0., 0.04])        
        y_lo, y_hi = ax.get_ylim()
        verts = [(5, y_lo), (5, y_hi), (10, y_hi), (10, y_lo)]
        poly = Polygon(verts, facecolor='0.95', edgecolor='0.95')
        ax.add_patch(poly)
    else:
        _plt.ylim(-0.05, 0.08)
        _plt.yticks([-0.04, 0, 0.04, 0.08])


    mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
    mF.setTicksAndLims(xlabel="epochs", ylabel="$\\Delta$ width of\n%d%% HPD" % int(pct*100), xticks=range(0, lastDecodedEp+1, xticksEvry), yticksD=None, xlim=[1-0.3, lastDecodedEp+0.3], tickFS=tkFS, labelFS=lbFS)


    fig.subplots_adjust(left=0.27, bottom=0.14, right=0.99, top=0.98, wspace=0.3, hspace=0.3)
    _plt.savefig("cmpscrs_%s.png" % anim, transparent=True)
    _plt.close()

#################3  summary
###########  r2, c95, w95
fig  = _plt.figure(figsize=(6, 8))
r2a  = _N.mean(r2as, axis=0)
r2e  = _N.mean(r2es, axis=0)
w95a = _N.mean(w95as, axis=0)
w95e = _N.mean(w95es, axis=0)
c95a = _N.mean(c95as, axis=0)
c95e = _N.mean(c95es, axis=0)
##################################################  (3, 1, 1)
ax   = fig.add_subplot(3, 1, 1)
_plt.errorbar(range(1, lastDecodedEp+1), r2a, yerr=r2e, fmt='o', ls="-", color=plot_clr, lw=2)
_plt.axhline(y=0, ls=":", color="grey")

if nhists >= 1:
    r2_1s13a = _N.mean(r2_1s13as)
    r2_1s13e = _N.mean(r2_1s13es)
    _plt.errorbar([lastDecodedEp-0.19, lastDecodedEp-0.19], [r2_1s13a, r2_1s13a], yerr=r2_1s13e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
if nhists >= 2:
    r2_1s11a = _N.mean(r2_1s11as)
    r2_1s11e = _N.mean(r2_1s11es)
    _plt.errorbar([lastDecodedEp-2-0.19, lastDecodedEp-2-0.19], [r2_1s11a, r2_1s11a], yerr=r2_1s11e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
if nhists >= 3:
    r2_1s9a = _N.mean(r2_1s9as)
    r2_1s9e = _N.mean(r2_1s9es)
    _plt.errorbar([lastDecodedEp-4-0.19, lastDecodedEp-4-0.19], [r2_1s9a, r2_1s9a], yerr=r2_1s9e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
if nhists >= 4:
    r2_1s7a = _N.mean(r2_1s7as)
    r2_1s7e = _N.mean(r2_1s7es)
    _plt.errorbar([lastDecodedEp-6-0.19, lastDecodedEp-6-0.19], [r2_1s7a, r2_1s7a], yerr=r2_1s7e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
if nhists >= 5:
    r2_1s5a = _N.mean(r2_1s5as)
    r2_1s5e = _N.mean(r2_1s5es)
    _plt.errorbar([lastDecodedEp-8-0.19, lastDecodedEp-8-0.19], [r2_1s5a, r2_1s5a], yerr=r2_1s5e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
    _plt.ylim(-0.2, 0.8)
    _plt.yticks([0, 0.4, 0.8])
    y_lo, y_hi = ax.get_ylim()
    verts = [(5, y_lo), (5, y_hi), (10, y_hi), (10, y_lo)]
    poly = Polygon(verts, facecolor='0.95', edgecolor='0.95')
    ax.add_patch(poly)
else:
    _plt.ylim(-0.2, 0.8)

mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
mF.setTicksAndLims(ylabel="$\Delta$ rMSE", xticks=range(0, lastDecodedEp+1, xticksEvry), yticksD=None, xlim=[1-0.3, lastDecodedEp+0.3], tickFS=tkFS, labelFS=lbFS)

##################################################  (3, 1, 2)
ax = fig.add_subplot(3, 1, 2)

_plt.axhline(y=0, ls=":", color="grey")
_plt.errorbar(range(1, lastDecodedEp+1), c95a, yerr=c95e, fmt='o', ls="-", color=plot_clr, lw=2)
if nhists >= 1:
    c95_1s13a = _N.mean(c95_1s13as)
    c95_1s13e = _N.mean(c95_1s13es)
    _plt.errorbar([lastDecodedEp-0.19, lastDecodedEp-0.19], [c95_1s13a, c95_1s13a], yerr=c95_1s13e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
if nhists >= 2:
    c95_1s11a = _N.mean(c95_1s11as)
    c95_1s11e = _N.mean(c95_1s11es)
    _plt.errorbar([lastDecodedEp-2-0.19, lastDecodedEp-2-0.19], [c95_1s11a, c95_1s11a], yerr=c95_1s11e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
if nhists >= 3:
    c95_1s9a = _N.mean(c95_1s9as)
    c95_1s9e = _N.mean(c95_1s9es)
    _plt.errorbar([lastDecodedEp-4-0.19, lastDecodedEp-4-0.19], [c95_1s9a, c95_1s9a], yerr=c95_1s9e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
if nhists >= 4:
    c95_1s7a = _N.mean(c95_1s7as)
    c95_1s7e = _N.mean(c95_1s7es)
    _plt.errorbar([lastDecodedEp-6-0.19, lastDecodedEp-6-0.19], [c95_1s7a, c95_1s7a], yerr=c95_1s7e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
if nhists >= 5:
    c95_1s5a = _N.mean(c95_1s5as)
    c95_1s5e = _N.mean(c95_1s5es)
    _plt.errorbar([lastDecodedEp-8-0.19, lastDecodedEp-8-0.19], [c95_1s5a, c95_1s5a], yerr=c95_1s5e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)

    _plt.ylim(-0.15, 0.03)
    _plt.yticks([-0.1, -0.05, 0., ])    
    y_lo, y_hi = ax.get_ylim()
    verts = [(5, y_lo), (5, y_hi), (10, y_hi), (10, y_lo)]
    poly = Polygon(verts, facecolor='0.95', edgecolor='0.95')
    ax.add_patch(poly)
else:
    _plt.ylim(-0.15, 0.05)



mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
mF.setTicksAndLims(ylabel="$\\Delta$%% TU%dHPD" % int(pct*100), xticks=range(0, lastDecodedEp+1, xticksEvry), yticksD=None, xlim=[1-0.3, lastDecodedEp+0.3], tickFS=tkFS, labelFS=lbFS)

ax = fig.add_subplot(3, 1, 3)
_plt.axhline(y=0, ls=":", color="grey")

_plt.errorbar(range(1, lastDecodedEp+1), w95a, yerr=w95e, fmt='o', ls="-", color=plot_clr, lw=2)
if nhists >= 1:
    w95_1s13a = _N.mean(w95_1s13as)
    w95_1s13e = _N.mean(w95_1s13es)
    _plt.errorbar([lastDecodedEp-0.19, lastDecodedEp-0.19], [w95_1s13a, w95_1s13a], yerr=w95_1s13e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
if nhists >= 2:
    w95_1s11a = _N.mean(w95_1s11as)
    w95_1s11e = _N.mean(w95_1s11es)
    _plt.errorbar([lastDecodedEp-2-0.19, lastDecodedEp-2-0.19], [w95_1s11a, w95_1s11a], yerr=w95_1s11e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
if nhists >= 3:
    w95_1s9a = _N.mean(w95_1s9as)
    w95_1s9e = _N.mean(w95_1s9es)
    _plt.errorbar([lastDecodedEp-4-0.19, lastDecodedEp-4-0.19], [w95_1s9a, w95_1s9a], yerr=w95_1s9e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
if nhists >= 4:
    w95_1s7a = _N.mean(w95_1s7as)
    w95_1s7e = _N.mean(w95_1s7es)
    _plt.errorbar([lastDecodedEp-6-0.19, lastDecodedEp-6-0.19], [w95_1s7a, w95_1s7a], yerr=w95_1s7e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)
if nhists >= 5:
    w95_1s5a = _N.mean(w95_1s5as)
    w95_1s5e = _N.mean(w95_1s5es)
    _plt.errorbar([lastDecodedEp-8-0.19, lastDecodedEp-8-0.19], [w95_1s5a, w95_1s5a], yerr=w95_1s5e, fmt='o', ls="-", color="red", lw=1, marker="^", ms=9)

    _plt.ylim(-0.02, 0.04)
    _plt.yticks([-0.02, 0., 0.02, 0.04])        
    y_lo, y_hi = ax.get_ylim()
    verts = [(5, y_lo), (5, y_hi), (10, y_hi), (10, y_lo)]
    poly = Polygon(verts, facecolor='0.95', edgecolor='0.95')
    ax.add_patch(poly)
else:
    _plt.ylim(-0.05, 0.08)
    _plt.yticks([-0.04, 0, 0.04])


mF.arbitraryAxes(ax, axesVis=[True, True, False, False], xtpos="bottom", ytpos="left")
mF.setTicksAndLims(xlabel="epochs", ylabel="$\\Delta$ width of\n%d%% HPD" % int(pct*100), xticks=range(0, lastDecodedEp+1, xticksEvry), yticksD=None, xlim=[1-0.3, lastDecodedEp+0.3], tickFS=tkFS, labelFS=lbFS)


fig.subplots_adjust(left=0.27, bottom=0.14, right=0.99, top=0.98, wspace=0.3, hspace=0.3)
_plt.savefig("cmpscrs_all.png", transparent=True)
_plt.savefig("cmpscrs_all.eps", transparent=True)
_plt.close()


