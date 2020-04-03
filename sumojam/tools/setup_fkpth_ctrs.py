import numpy as _N
import matplotlib.pyplot as _plt

#segname = "fake_md2d0102_3"
segname = "remy_md2d0102_4"
#segname = "remy_6arm0102_4"
d = _N.loadtxt("/Users/arai/nctc/Workspace/jifimogs/DATA/%s.dat" % segname)
#d = _N.random.multivariate_normal(_N.array([0, 0]), _N.array([[1., 0.], [0., 1.]]), size=1000)

iSeps    = []    # separator location
xy_dats = []
iG      = 0
Nt      = 1000000.   #  total number of points

def OnRelease(event):  #####  add fake path cluster 
    global iG, xy_dats
    print("DOWN    ", event)
    xy_dats.append([event.xdata, event.ydata])
    _plt.plot([event.xdata, event.xdata], [event.ydata, event.ydata], marker=".", ms=4, color="black")
    iG += 1

def KeyPress(event):
    global iG, iSeps, xy_dats
    print("KEY    ", event)
    if event.key == 'd':   #  saving into .prms
        mns      = _N.array(xy_dats)
        xy_pts = _N.ones((mns.shape[0], 5))
        xy_pts[:, 0:2] = _N.array(xy_dats)
        xy_pts[:, 2]   = 64
        xy_pts[:, 3]   = 64
        xy_pts[:, 4]   =  (Nt / mns.shape[0])
        
        fp = open("/Users/arai/nctc/Workspace/jifimogs/DATA/%s_fkpth.prms" % segname, "w")
        iS = 0
        for i in range(iG):
            fp.write("%(1).3e %(2).3e %(3).3e %(4).3e %(5).3e\n" % {"1" : xy_pts[i, 0], "2" : xy_pts[i, 1], "3" : xy_pts[i, 2], "4" : xy_pts[i, 3], "5" : xy_pts[i, 4]})
            if (iS < len(iSeps)) and (i == iSeps[iS] - 1):
                iS += 1
                fp.write("############################\n")
        fp.close()

    if event.key == 'b':   #  adding a separator for easy file modification 
        iSeps.append(iG)
        

fig  = _plt.figure(figsize=(8, 8))
_plt.plot(d[::50, 0], d[::50, 1])
xlo = _N.min(d[:, 0])
xhi = _N.max(d[:, 0])
ylo = _N.min(d[:, 1])
yhi = _N.max(d[:, 1])
xA  = xhi-xlo
yA  = yhi-ylo

_plt.xlim(xlo - 0.3*xA, xhi + 0.3*xA)
_plt.ylim(ylo - 0.3*yA, yhi + 0.3*yA)

cid_down = fig.canvas.mpl_connect('button_release_event', OnRelease)
kyd_down = fig.canvas.mpl_connect('key_press_event', KeyPress)
