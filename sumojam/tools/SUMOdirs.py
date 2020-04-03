import os

"""
def resFN(fn, dir=None, create=False):
    #  ${JIFIdir}/Results/
    __JIFIResultDir__ = os.environ["__JIFIResultDir__"]
    rD = __JIFIResultDir__

    if dir != None:
        rD = "%(rd)s/%(ed)s" % {"rd" : __JIFIResultDir__, "ed" : dir}
        if not os.access("%s" % rD, os.F_OK) and create:
            os.mkdir(rD)
    return "%(rd)s/%(fn)s" % {"rd" : rD, "fn" : fn}
"""

def recurseMkdir(dirstr):
    lvls = dirstr.split("/")
    rD   = ""
    for lvl in lvls:
        rD += "%s/" % lvl
        if not os.access("%s" % rD, os.F_OK):
            os.mkdir(rD)

def resFN(fn, __JIFIResultDir__=None, dir=None, create=False):
    #  ${JIFIdir}/Results/
    __JIFIResultDir__ = os.environ["__JIFIResultDir__"] if __JIFIResultDir__ is None else __JIFIResultDir__
    rD = __JIFIResultDir__

    if dir != None:
        lvls = dir.split("/")
        for lvl in lvls:
            rD += "/%s" % lvl
            if not os.access("%s" % rD, os.F_OK) and create:
                os.mkdir(rD)
    return "%(rd)s/%(fn)s" % {"rd" : rD, "fn" : fn}

def pracFN(fn, dir=None, create=False):
    #  ${JIFIdir}/Results/
    __JIFIPracDir__ = os.environ["__JIFIPracDir__"]
    rD = __JIFIPracDir__

    if dir != None:
        lvls = dir.split("/")
        for lvl in lvls:
            rD += "/%s" % lvl
            if not os.access("%s" % rD, os.F_OK) and create:
                os.mkdir(rD)
    return "%(rd)s/%(fn)s" % {"rd" : rD, "fn" : fn}

def prcmpFN(fn, dir=None, create=False):
    #  ${JIFIdir}/Results/
    __JIFIPrecompDir__ = os.environ["__JIFIPrecompDir__"]
    pD = __JIFIPrecompDir__

    if dir != None:
        pD = "%(rd)s/%(ed)s" % {"rd" : __JIFIPrecompDir__, "ed" : dir}
        if not os.access("%s" % pD, os.F_OK) and create:
            os.mkdir(pD)
    return "%(rd)s/%(fn)s" % {"rd" : pD, "fn" : fn}

def datFN(fn, dir=None, create=False):
    #  ${JIFIdir}/Results/
    __JIFIDataDir__ = os.environ["__JIFIDataDir__"]
    dD = __JIFIDataDir__

    if dir != None:
        lvls = dir.split("/")
        for lvl in lvls:
            dD += "/%s" % lvl
            if not os.access("%s" % dD, os.F_OK) and create:
                os.mkdir(dD)
    return "%(dd)s/%(fn)s" % {"dd" : dD, "fn" : fn}


