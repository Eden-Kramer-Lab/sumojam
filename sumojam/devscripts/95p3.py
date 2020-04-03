import jifimogs.mkdecoder as _mkd
import jifimogs.tools.decodeutil as _du
import utilities as _U
import jifimogs.tools.decode_score as _d_s
import numpy as _N
from jifimogs.devscripts.cmdlineargs import process_keyval_args
import sys

def depickle(s):
     import pickle
     with open(s, "rb") as f:
          lm = pickle.load(f)
     return lm

#  CRCL
#  maximum distance one can be at is 6
#  5 and -5 is distance == 2   if dist > 6, then dist = 12-dist


_KDE_RT = True

pct        = 0.99
pctm1      = 1-pct
mode = 0
"""
##############################
anim       = "bond" # bn2
day        = "03"
ep         = "02"

##############################
anim       = "bond"
day        = "10"
ep         = "04"

##############################
anim       = "frank" # bn2
day        = "06"
ep         = "02"

##############################
anim       = "frank" # bn2
day        = "02"
ep         = "02"

##############################
anim       = "9bn"
day        = None
ep         = None
"""
anim       = "bc5"
day        = "01"
ep         = "01"

itvfn   = "itv2_15"
label     =  "1"   #  refers t5
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
        _tetslist.pop(16)
    elif andyep == "frank0602":
        _tetslist = list(range(100, 120))
        _tetslist.pop(17)        
        _tetslist.pop(15)
    elif (andyep == "noch0101") or (andyep == "bich0101"):
        _tetslist = list(range(11, 21))
    elif (andyep == "asch0101"):
        _tetslist = list(range(1, 7))    
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
    tet_or_sim_list = _N.arange(1, 6)



PD        = 12 if usemaze == _mkd.mz_CRCL else 6

lagfits    = [1,]
#lagfits    = [1,]

bx=0.15
Bx=0.15
Bm=4

ignr  = 0

for lagfit in lagfits:
    for tet_or_sim in tet_or_sim_list:
        if dat_mode == _du._TETS:
            tets = tet_or_sim
            print(tets)            
            datfns, resdns, outdir, tetstr = _du.makefilenames(anim, day, ep, tets, itvfn, label=label)            
        else:
            tets = None
            print(anim+str(tet_or_sim))
            datfns, resdns, outdir, tetstr = _du.makefilenames(anim+str(tet_or_sim), day, ep, tets, itvfn, label=label)                        

        #itvs      = _N.loadtxt(_edd.datFN("%s.dat" % itvfn))

        if method == _du._KDE:
            if _KDE_RT:
                outdirN = "%(d)s/decKDE_RT%(ts)s_%(lf)d_%(Bx).2f_%(Bm).2f_%(bx).2f" % {"d" : outdir, "ts" :tetstr, "lf" : lagfit, "Bx" : Bx, "Bm" : Bm, "bx" : bx}
            else:
                outdirN = "%(d)s/decKDE%(ts)s_%(lf)d" % {"d" : outdir, "ts" :tetstr, "lf" : lagfit}
        elif method == _du._MOG:
            outdirN = "%(d)s/decMoG%(ts)s_%(lf)d" % {"d" : outdir, "ts" :tetstr, "lf" : lagfit}
        else:
            outdirN = "%(d)s/decGT%(ts)s_%(lf)d" % {"d" : outdir, "ts" :tetstr, "lf" : lagfit}

        if method == _du._GT:
            meths = "GT"
        elif method == _du._KDE:
            meths = "KDE_RT"
        elif method == _du._MOG:
            meths = "MoG"    

        pcklfn = "%(od)s/pX_Nm.dmp" % {"od" : outdirN}
        _lm = depickle(pcklfn)
        lm  = _lm[1:]   #  new versions of pX_Nm.dmp have saveBin as 1st param

        #NBins=241
        NBins=121
        xp = _N.linspace(-6, 6, NBins)

        epts  = lm[4]


        if usemaze == _mkd.mz_W:
            sfmt   = "%.4f %4f %.4f %4f %.4f %4f"
            #  non-folded
            scrs = _N.empty((epts.shape[0]-2, 6))
            pX_Nm = lm[0]
            pos   = lm[1]

            _d_s.scores(pctm1, lagfit, epts, ignr, scrs[:, 0:3], xp, pX_Nm, pos, usemaze)

            ###  folding
            pX_Nm = lm[7]
            pos   = lm[6]
            #NBins = 61
            NBins = 31
            #xp    = _N.linspace(0, 3, 16)   #  6+5+5
            xp    = _N.linspace(0, 3, 31)   #  11+10+10
            #xp    = _N.linspace(0, 3, 61)    #  21+20+20
            _d_s.scores(pctm1, lagfit, epts, ignr, scrs[:, 3:6], xp, pX_Nm, pos, usemaze)
        else:
            sfmt   = "%.4f %4f %.4f"
            scrs = _N.empty((epts.shape[0]-2, 3))
            pX_Nm = lm[0]
            pos   = lm[1]
            _d_s.scores(pctm1, lagfit, epts, ignr, scrs, xp, pX_Nm, pos, usemaze)

        smth = _du._method_names[method]

        if method == _du._KDE:
            _U.savetxtWCom("%(od)s/%(sm)s_%(lf)d_%(pct).2f_scrs%(tet)s_%(Bx).2f_%(Bm).2f_%(bx).2f.txt" % {"od" : outdir, "sm" : smth, "tet" : str(tets).replace(" ", ",").replace("\n" ""), "lf" : lagfit, "Bx" : Bx, "Bm" : Bm, "bx" : bx, "pct" : pct}, scrs, fmt=sfmt, delimiter=" ", com="#  pct in 95th,  sz of 95, R2")
        else:
            _U.savetxtWCom("%(od)s/%(sm)s_%(lf)d_%(pct).2f_scrs%(tet)s.txt" % {"od" : outdir, "sm" : smth, "tet" : str(tets).replace(" ", ",").replace("\n", ""), "lf" : lagfit, "Bx" : Bx, "Bm" : Bm, "bx" : bx, "pct" : pct}, scrs, fmt=sfmt, delimiter=" ", com=("#  pct in %(pct).2f,  sz of %(pct).2f, R2" % {"pct" : pct}))
            print("output %(od)s/%(sm)s_%(lf)d_%(pct).2f_scrs%(tet)s.txt" % {"od" : outdir, "sm" : smth, "tet" : str(tets).replace(" ", ",").replace("\n", ""), "lf" : lagfit, "Bx" : Bx, "Bm" : Bm, "bx" : bx, "pct" : pct})

