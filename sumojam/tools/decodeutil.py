_KDE        = 0
_MOG        = 1
_GT         = 2

_TETS      = 0
_SIMS      = 1

_method_names= ["KDE", "MOG", "GT"]

#  read from DATA (to get new marks) and Results (of fit)
def makefilenames(anim, day, ep, tets, itvfn, label=None, ripple=False):
    #  basefilename for _prms.pkl
    #  directory name for Results/anim_
    #  outdir  = where to put decoded output.  "dec_animdyep"

    datfns = []
    resdns = []
    tetstr = None
    if tets is not None:   
        tetstr   = "_"
        slab     = ""
        if label is not None:
            slab = "-%s" % label

        for tet in tets:
            tetstr += "%d," % tet 
            if (day == None) or (ep == None):
                fn = anim
                dn = "%(a)s%(itv)s-%(lb)s" % {"a" : anim, "itv" : itvfn, "lb" : slab}
            else:
                itv = itvfn[3:]
                decmdec = "mdec" if not ripple else "dec"
                #if ripple:
                #fn = "%(a)s_%(md)s%(dyep)s_%(t)d" % {"a" : anim, "dyep" : (day+ep), "t" : tet, "md" : decmdec}
                fn = "%(a)s_%(md)s%(dyep)s_%(t)d" % {"a" : anim, "dyep" : (day+ep), "t" : tet, "md" : decmdec}
                #dn = "%(a)s_%(md)s%(dyep)s_%(t)d-%(itv)s" % {"a" : anim, "dyep" : (day+ep), "t" : tet, "itv" : itv, "md" : "mdec"}
                dn = "%(a)s_%(md)s%(dyep)s-%(itv)s%(lb)s/%(t)d" % {"a" : anim, "dyep" : (day+ep), "t" : tet, "itv" : itv, "md" : "mdec", "lb" : slab}
            datfns.append(fn)
            resdns.append(dn)
        tetstr = tetstr[:-1]

        outdir = "dec-%s" % "%(a)s%(dyep)s-%(itv)s" % {"a" : anim, "dyep" : (day+ep), "itv" : itv}
        if label is not None:
            outdir += "-%s" % label
    else:   #  simulation format
        slab     = ""
        if label is not None:
            slab = "-%s" % label

        datfns.append(anim)
        #resdns.append("%(a)s-%(itv)s%(lb)s" % {"a" : anim, "itv" : itvfn[3:]})#, "lb" : slab})
        resdns.append("%(a)s-%(itv)s%(lb)s" % {"a" : anim, "itv" : itvfn[3:], "lb" : slab})
        outdir = "dec-%s" % resdns[0]
        tetstr = ""


    return datfns, resdns, outdir, tetstr
