_KDE        = 0
_MOG        = 1
_GT         = 2

_TETS      = 0
_SIMS      = 1

_method_names= ["KDE", "MOG", "GT"]

#  read from DATA (to get new marks) and Results (of fit)
def makefilenames(anim, day, ep, tets, itvfn, label, mode, dim=1, ripple=False):
    #  basefilename for _prms.pkl
    #  directory name for Results/anim_
    #  outdir  = where to put decoded output.  "dec_animdyep"

    datfns = []
    resdns = []

    tetstr=""
    if type(tets) is not list:
        tets = [tets]
    for tet in tets:
        tetstr += "%d," % tet 

        itv = itvfn[3:]
        if dim == 1:
            decmdec = "mdec"
        else:
            decmdec = "md2d"
        fn = "%(a)s_%(md)s%(dyep)s_%(t)d" % {"a" : anim, "dyep" : (day+ep), "t" : tet, "md" : decmdec}
        dn = "%(a)s/%(md)s%(dyep)s/%(itv)s-%(lb)s/%(t)d" % {"a" : anim, "dyep" : (day+ep), "t" : tet, "itv" : itv, "md" : decmdec, "lb" : label}
        datfns.append(fn)
        resdns.append(dn)

    tetstr = tetstr[:-1]   #  last comma

    outdir = "%(a)s/%(md)s%(dyep)s/%(itv)s-%(lb)s/dec/%(t)s" % {"a" : anim, "dyep" : (day+ep), "t" : tetstr, "itv" : itv, "md" : decmdec, "lb" : label}

    return datfns, resdns, outdir, tetstr
