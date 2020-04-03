import jifimogs.tools.EnDedirs as _edd
import numpy as _N


def slopes_of_segs(segs):
    #  a_s  slope
    a_s     = (segs[:, 1, 1] - segs[:, 0, 1]) / (segs[:, 1, 0] - segs[:, 0, 0]) 
    b_s     = -1
    c_s     = segs[:, 0, 1] - a_s*segs[:, 0, 0] 
    return a_s, b_s, c_s


#def find_clsest(long n, double x0, double y0, double *p_segs, double *p_dists, long *p_seg_ts, long Nsgs, char *p_online, double *p_xcs, double *p_ycs, mins):
def find_clsest(long n, double x0, double y0, double[:, :, ::1] mv_segs, double[:, ::1] mv_dists, long[::1] mv_seg_ts, long Nsgs, char[::1] mv_online, double[:, ::1] mv_xcs, double[:, ::1] mv_ycs, mins):
    # Find the closest segment to point (rawpos) x0, y0
    cdef double *p_xcs = &mv_xcs[0, 0], *p_ycs = &mv_ycs[0, 0]
    cdef char *p_online= &mv_online[0]
    cdef long* p_seg_ts = &mv_seg_ts[0]
    cdef double* p_segs = &mv_segs[0, 0, 0]
    cdef double* p_dists = &mv_dists[0, 0]

    cdef double x1, y1, x2, y2
    cdef nNsgs = n * Nsgs

    cdef double[::1] mv_mins = mins
    cdef double* p_mins = &mv_mins[0]
    cdef double onl_tmp

    
    for ns in range(Nsgs):
        #  x-coords at segs[ns, 0, 0], segs[ns, 1, 0]
        #  y-coords at segs[ns, 0, 1], segs[ns, 1, 1]
        x1  = p_segs[4*ns]
        y1  = p_segs[4*ns+1]
        x2  = p_segs[4*ns+2]
        y2  = p_segs[4*ns+3]
        #  p_xcs[nNsgs+ns] = xcs[n, ns]
        """
        onlinex = (((p_xcs[nNsgs+ns] >= x1) and (p_xcs[nNsgs+ns] <= x2)) or 
                   ((p_xcs[nNsgs+ns] <= x1) and (p_xcs[nNsgs+ns] >= x2)))
        onliney = (((p_ycs[nNsgs+ns] >= y1) and (p_ycs[nNsgs+ns] <= y2)) or 
                   ((p_ycs[nNsgs+ns] <= y1) and (p_ycs[nNsgs+ns] >= y2)))
        p_online[ns] = onlinex and onliney  #  closest point is on line
        """
        p_online[ns] = (((p_xcs[nNsgs+ns] >= x1) and (p_xcs[nNsgs+ns] <= x2) and
                         (p_ycs[nNsgs+ns] >= y1) and (p_ycs[nNsgs+ns] <= y2))
                        or
                        ((p_xcs[nNsgs+ns] >= x2) and (p_xcs[nNsgs+ns] <= x1) and
                         (p_ycs[nNsgs+ns] >= y1) and (p_ycs[nNsgs+ns] <= y2))
                        or
                        ((p_xcs[nNsgs+ns] >= x1) and (p_xcs[nNsgs+ns] <= x2) and
                         (p_ycs[nNsgs+ns] >= y2) and (p_ycs[nNsgs+ns] <= y1))
                        or
                        ((p_xcs[nNsgs+ns] >= x2) and (p_xcs[nNsgs+ns] <= x1) and
                         (p_ycs[nNsgs+ns] >= y2) and (p_ycs[nNsgs+ns] <= y1)))


        
        #  I never use mins
        p_mins[ns]  = p_dists[n*2*Nsgs + ns] if p_dists[n*2*Nsgs + ns] < p_dists[n*2*Nsgs + ns + 1] else p_dists[n*2*Nsgs + ns + 1]
        onl_tmp  = (x0-p_xcs[nNsgs+ns])**2 + (y0-p_ycs[nNsgs+ns])**2  #  this should be it.
        p_mins[ns] = onl_tmp if onl_tmp < p_mins[ns] else p_mins[ns]


        """        
        if p_online[ns]:   #  closest point (x0, y0) on segment, not endpts
            #  min(1 number, min(array))
            #mins[ns] = _N.min([(x0-p_xcs[ns])**2 + (y0-p_ycs[ns])**2, _N.min(rdists[n, ns])])  #  if p_online[ns], the closest point should always be xcs, ycs
            p_mins[ns] = (x0-p_xcs[nNsgs+ns])**2 + (y0-p_ycs[nNsgs+ns])**2  #  this should be it.
            onl_tmp  = (x0-p_xcs[nNsgs+ns])**2 + (y0-p_ycs[nNsgs+ns])**2  #  this should be it.
            #p_mins[ns] = onl_tmp if onl_tmp < p_mins[ns] else p_mins[ns]
            #  this should be it.

        else:
            #mins[ns] = _N.min(rdists[n, ns])  #  either end point
            #  dists[n, ns, 0]
            p_mins[ns]  = p_dists[n*2*Nsgs + ns] if p_dists[n*2*Nsgs + ns] < p_dists[n*2*Nsgs + ns + 1] else p_dists[n*2*Nsgs + ns + 1]
                
            #mins[ns] = _N.min(rdists[n, ns])  #  either end point
        """

    clsest = _N.where(mins == _N.min(mins))[0]
    iclsest= clsest[0]        #  segment ID that is closest to x0, y0
    p_seg_ts[n] = clsest[0]
    return iclsest


def on_segs_or_off(double nz_scl, long N1kHz, double[::1] mv_x0s, double[::1] mv_y0s, unit_norm_x, unit_norm_y, double[:, :, ::1] mv_segs, double[:, ::1] mv_dists, long[::1] mv_seg_ts, long Nsgs, double[:, ::1] mv_xcs, double[:, ::1] mv_ycs):
    """
    1) index of closest segment           
    2) whether closest point is on the segment  (online)
    """

    online = _N.zeros(Nsgs, dtype=_N.uint8)   #  closest point on segment?
    cdef double *p_x0s = &mv_x0s[0], *p_y0s = &mv_y0s[0]
    cdef char[::1] mv_online = online
    cdef char *p_online= &mv_online[0]
    cdef double *p_xcs = &mv_xcs[0, 0], *p_ycs = &mv_ycs[0, 0]
    cdef double* p_segs = &mv_segs[0, 0, 0]
    cdef long* p_seg_ts = &mv_seg_ts[0]
    """
    cdef char *p_online= &mv_online[0]
    cdef double x0, y0
    cdef int i_clsest

    cdef double* p_dists = &mv_dists[0, 0]
    """

    mins = _N.empty(Nsgs)

    cdef long n

    rn_xs = _N.random.randn(N1kHz)
    rn_ys = _N.random.randn(N1kHz)
    cdef double[::1] mv_rn_xs = rn_xs
    cdef double[::1] mv_rn_ys = rn_ys
    cdef double *p_rn_xs      = &mv_rn_xs[0]
    cdef double *p_rn_ys      = &mv_rn_ys[0]
    cdef double[::1] mv_unit_norm_x = unit_norm_x
    cdef double[::1] mv_unit_norm_y = unit_norm_y
    cdef double *p_unit_norm_x      = &mv_unit_norm_x[0]
    cdef double *p_unit_norm_y      = &mv_unit_norm_y[0]

    on_ts = []
    off_ts     = []
    allpts       = []
    
    for n in range(N1kHz):  #  not spike times
        x0 = p_x0s[n]
        y0 = p_y0s[n]
        #i_clsest = find_clsest(n, x0, y0, p_segs, p_dists, p_seg_ts, Nsgs, p_online, p_xcs, p_ycs, mins)

        i_clsest = find_clsest(n, x0, y0, mv_segs, mv_dists, mv_seg_ts, Nsgs, mv_online, mv_xcs, mv_ycs, mins)

        if len(_N.where(online == 1)[0]) > 0:
            r = nz_scl*p_rn_xs[n]
            nc = p_seg_ts[n]    #  index of closest segment
            #  if you want to see assignment of raw data to seg or off-seg
            #nzx = x0#
            #nzy = y0#

            nzx = p_xcs[n*Nsgs + nc] + r * p_unit_norm_x[nc]
            nzy = p_ycs[n*Nsgs + nc] + r * p_unit_norm_y[nc]

            on_ts.append(n)
        else:
            nzx = x0 + nz_scl*p_rn_xs[n]
            nzy = y0 + nz_scl*p_rn_ys[n]

            off_ts.append(n)
            
        allpts.append([nzx, nzy])

    return on_ts, off_ts, _N.array(allpts)
