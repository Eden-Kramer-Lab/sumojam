# 2018.12.19 01:28:13 EST
#Embedded file name: manypgs.py
import matplotlib.pyplot as _plt

class manyfigsPP:
    ax = None
    rows = None
    cols = None
    fig = None
    W = None
    H = None
    pn = 0
    pg = 0
    sbp_left = 0.15
    sbp_right = 0.95
    sbp_bottom = 0.15
    sbp_top = 0.95
    sbp_wspace = 0.2
    sbp_hspace = 0.2
    base_filename = None
    savedir = None

    def __init__(self, bfn, r = 1, c = 1, W = 5, H = 4, savedir = None, left = 0.15, bottom = 0.15, top = 0.95, right = 0.95, wspace = 0.2, hspace = 0.2, suptitle=None):
        oo = self
        oo.rows = r
        oo.cols = c
        oo.W = W
        oo.H = H
        oo.base_filename = bfn
        oo.savedir = savedir
        oo.sbp_left = left
        oo.sbp_right = right
        oo.sbp_bottom = bottom
        oo.sbp_top = top
        oo.sbp_wspace = wspace
        oo.sbp_hspace = hspace
        oo.suptitle   = suptitle

    def addplot(self, title = None, fontsize = 12):
        oo = self
        if oo.fig is None:
            oo.fig = _plt.figure(figsize=(oo.W, oo.H))
            oo.pn = 1
            oo.pg += 1
            if oo.suptitle is not None:
                _plt.suptitle(oo.suptitle)
        oo.ax = oo.fig.add_subplot(oo.rows, oo.cols, oo.pn)
        if title is not None:
            _plt.title(title, fontsize=fontsize)
        oo.pn += 1

    def doneplot(self):
        oo = self
        if oo.pn == oo.rows * oo.cols + 1:
            oo.save()

    def save(self):
        oo = self
        if oo.fig is not None:
            fn = '%(fn)s_pg%(pg)d' % {'fn': oo.base_filename,
             'pg': oo.pg}
            if oo.savedir is not None:
                fn = oo.savedir + '/' + fn
            oo.fig.subplots_adjust(left=oo.sbp_left, right=oo.sbp_right, bottom=oo.sbp_bottom, top=oo.sbp_top, wspace=oo.sbp_wspace, hspace=oo.sbp_hspace)
            print(fn)
            _plt.savefig(fn)
            _plt.close()
            oo.fig = None
#+++ okay decompyling manypgs.pyc 
# decompiled 1 files: 1 okay, 0 failed, 0 verify failed
# 2018.12.19 01:28:13 EST
