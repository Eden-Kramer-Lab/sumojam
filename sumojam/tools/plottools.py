import numpy as _N
import matplotlib.pyplot as _plt


def setTicksAndLims(xlabel=None, ylabel=None, xticks=None, yticks=None, xticksD=None, yticksD=None, xlim=None, ylim=None, tickFS=26, labelFS=28):
    if xticks is not None:
        if xticksD is None:
            _plt.xticks(xticks, fontsize=tickFS)
        else:
            _plt.xticks(xticks, xticksD, fontsize=tickFS)
    else:
        _plt.xticks(fontsize=tickFS)
    if yticks is not None:
        if yticksD is None:
            _plt.yticks(yticks, fontsize=tickFS)
        else:
            _plt.yticks(yticks, yticksD, fontsize=tickFS)
    else:
        _plt.yticks(fontsize=tickFS)

    if xlim is not None:
        if type(xlim) == list:
            _plt.xlim(xlim[0], xlim[1])
        else:
            _plt.xlim(0, xlim)
    if ylim is not None:
        if type(ylim) == list:
            _plt.ylim(ylim[0], ylim[1])
        else:
            _plt.ylim(0, ylim)

    if xlabel is not None:
        _plt.xlabel(xlabel, fontsize=labelFS)
    if ylabel is not None:
        _plt.ylabel(ylabel, fontsize=labelFS)

def bottomLeftAxes(ax, bottomVis=True, leftVis=True):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(bottomVis)
    ax.spines["left"].set_visible(leftVis)

    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["right"].axis.axes.tick_params(direction="outward", width=2)
    ax.spines["top"].axis.axes.tick_params(direction="outward", width=2)

def vstackedPlots(ax, bottom=False):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if not bottom:
        ax.spines["bottom"].set_visible(False)
        _plt.xticks([])
    else:
        ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    ax.spines["right"].axis.axes.tick_params(direction="outward", width=2)
    if bottom:
        ax.spines["top"].axis.axes.tick_params(direction="outward", width=2)

def arbitraryAxes(ax, axesVis=[True, True, True, True], xtpos="bottom", ytpos="left"):
    """
    Left, Bottom, Right, Top        Axes line itself
    xtpos               "bottom", "top", "both", "none"
    "left", "right", "both", "none"
    """
    ax.spines["left"].set_visible(axesVis[0])
    ax.spines["bottom"].set_visible(axesVis[1])
    ax.spines["right"].set_visible(axesVis[2])
    ax.spines["top"].set_visible(axesVis[3])

    ax.xaxis.set_ticks_position(xtpos)
    ax.yaxis.set_ticks_position(ytpos)
    ax.xaxis.set_label_position(xtpos)
    ax.yaxis.set_label_position(ytpos)

    #ax.spines["left"].axis.axes.tick_params(direction="inward", width=2)
    #ax.spines["bottom"].axis.axes.tick_params(direction="outward", width=2)
    #ax.spines["right"].axis.axes.tick_params(direction="outward", width=2)
    #ax.spines["top"].axis.axes.tick_params(direction="outward", width=2)
    ax.spines["left"].axis.axes.tick_params(direction="in", width=2)
    ax.spines["bottom"].axis.axes.tick_params(direction="out", width=2)
    ax.spines["right"].axis.axes.tick_params(direction="out", width=2)
    ax.spines["top"].axis.axes.tick_params(direction="out", width=2)


def int_bins_4_ticks(dat, nbins):
    """
    choose good bins at integer locations 
    xlo, xhi are 
    """
    xlo = int(_N.floor(_N.min(dat)))
    xhi = int(_N.ceil(max(dat)))

    xA  = xhi - xlo

    remainder = xA % nbins

    if remainder // 2 == 0:
        xlo -= remainder/2
        xhi += remainder/2
    else:
        xlo -= remainder/2+1
        xhi += remainder/2

    return _N.linspace(xlo, xhi, nbins, endpoint=True)
