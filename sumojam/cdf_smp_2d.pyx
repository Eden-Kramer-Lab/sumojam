import numpy as _N
cimport numpy as _N
import cython
cimport cython
import time as _tm
import utilities as _U
from libc.math cimport exp, sqrt, log
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
import matplotlib.pyplot as _plt
#uniqFN(filename, serial=False, iStart=1, returnPrevious=False)
import utilities as _U
import scipy.stats as _ss

twoPOW = _N.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], dtype=_N.int)
cdef long[::1] v_twoPOW = twoPOW
cdef long* p_twoPOW = &v_twoPOW[0]
cdef double x_Lo, x_Hi, y_Lo, y_Hi

cdef int __NRM = 0
cdef int __IG  = 1

cdef double fourpi2 = 39.47841760435743
cdef double twpi = 6.283185307179586

# cdef double[::1] v_cpf2
# cdef double[::1] v_cpq22
# cdef double[::1] v_fx2
# cdef double[::1] v_qx2
cdef double *p_cpf2
cdef double *p_cpq22
cdef double *p_fx2
cdef double *p_qx2

#cdef unsigned char[::1] v_q2_smpld_support
#cdef unsigned char[::1] v_f_smpld_support
cdef unsigned char *p_q2_smpld_support
cdef unsigned char *p_f_smpld_support

cdef long *p_adtvInds

#####  variables for changing resolution of occupation function
#  riemann_xys:  sum over path replaced by occupation weighted sum over grid
cdef double *p_riemann_xs
cdef double *p_hist_all
cdef double *p_ibinszs
cdef double *p_q2_thr
cdef double *p_p
cdef double *p_dx
cdef double *p_cdf
cdef int minSmps
cdef double isqrt_twpi

cdef long *Nupxs
cdef long *arr_0s
cdef long n_res   #  number of different resolutions 
cdef double x_lo, x_hi

#cdef double x_lo, x_hi

cdef int f_STEPS   = 0
cdef int q2_STEPS  = 0
cdef int f_cldz    = 0
cdef int q2_cldz   = 0
cdef int f_SMALL   = 10
cdef int q2_SMALL  = 10


CDF_GRID_SIZE = 2000   #  big enough

q2_smpd_support = None   #  size 2**q2_STEPS+1
f_smpd_support  = None   #  size 2**f_STEPS+1
adtvInds        = None
cdf             = None
dx              = None
p               = None

cpf2 = None    # value of cond. post, defined only certain pts
cpq22 = None  

_NRM = 0   #  for outside visibility
_IG  = 1

fx2 = None    # grid where cond posterior function is defined and adptvly smpld
qx2 = None    # grid where cond posterior function is defined and adptvly smpld


f_lo = None
f_hi = None
q2_lo = None
q2_hi = None
cdef double dt

adtv_pdf_params = _N.empty(2)

########################################################################
def init(_dt, _f_lo, _f_hi, _q2_lo, _q2_hi, _f_STEPS, _q2_STEPS, _f_SMALL, _q2_SMALL, _f_cldz, _q2_cldz, _minSmps):
    """
    init the grid used to sample the conditional posterior functions
    unless maze is highly rectangular, _f_lo, _f_hi being set same for both 
    x and y direction not a problem
    """
    global fx2, qx2, cpf2, cpq22, f_smpld_support, q2_smpld_support, 
    global adtvInds, p_adtvInds, p_f_smpld_support, p_q2_smpld_support
    global p_fx2, p_qx2, p_cpf2, p_cpq22, 
    global dt
    global p_cdf, p_p, p_dx    
    global fx_lo, fx_hi, fy_lo, fy_hi, q2_lo, q2_hi
    global f_SMALL, q2_SMALL, f_STEPS, q2_STEPS, f_cldz, q2_cldz
    global minSmps
    global CDF_GRID_SIZE
    global cdf, p, dx
    global twpi, isqrt_twpi

    cdef unsigned char[::1] v_f_smpld_support
    cdef unsigned char[::1] v_q2_smpld_support
    cdef long[::1] v_adtvInds
    
    isqrt_twpi = 1./sqrt(twpi)

    minSmps   = _minSmps
    f_SMALL   = _f_SMALL
    q2_SMALL  = _q2_SMALL

    f_STEPS   = _f_STEPS
    q2_STEPS  = _q2_STEPS

    f_cldz    = _f_cldz
    q2_cldz   = _q2_cldz

    f_lo      = _f_lo
    f_hi      = _f_hi
    q2_lo     = _q2_lo
    q2_hi     = _q2_hi

    fx2 = _N.linspace(f_lo, f_hi, 2**f_STEPS+1, endpoint=True)
    qx2 = _N.exp(_N.linspace(_N.log(q2_lo), _N.log(q2_hi), 2**q2_STEPS+1, endpoint=True))
    cpf2 = _N.empty(2**f_STEPS+1)
    cpq22= _N.empty(2**f_STEPS+1)
    cdef double[::1]  v_fx2, v_qx2, v_cpf2, v_cpq22
    v_fx2 = fx2
    v_qx2 = qx2
    v_cpf2= cpf2
    v_cpq22=cpq22
    
    p_fx2  = &v_fx2[0]
    p_qx2  = &v_qx2[0]
    p_cpf2 = &v_cpf2[0]
    p_cpq22= &v_cpq22[0]    

    f_smpld_support = _N.zeros(2**f_STEPS+1, dtype=_N.uint8)
    q2_smpld_support= _N.zeros(2**q2_STEPS+1, dtype=_N.uint8)
    adtvInds        = _N.zeros(2**q2_STEPS+1, dtype=_N.int)
    cdf  = _N.zeros(CDF_GRID_SIZE)
    p    = _N.zeros(CDF_GRID_SIZE)
    dx   = _N.zeros(CDF_GRID_SIZE)    
    cdef double[::1] v_cdf  = cdf
    cdef double[::1] v_p    = p
    cdef double[::1] v_dx   = dx
    p_p    = &v_p[0]
    p_cdf  = &v_cdf[0]
    p_dx   = &v_dx[0]        

    v_fx2     = fx2
    v_qx2     = qx2
    v_cpf2    = cpf2
    v_cqf22   = cpq22

    v_f_smpld_support  = f_smpld_support
    v_q2_smpld_support = q2_smpld_support
    p_f_smpld_support  = &v_f_smpld_support[0]
    p_q2_smpld_support = &v_q2_smpld_support[0]    
    v_adtvInds         = adtvInds
    p_adtvInds         = &v_adtvInds[0]

    dt        = _dt

    ########################################################################
@cython.cdivision(True)
def l0_spatial(long M, int totalpcs, double dt, double[::1] ap_Ns, double[:, ::1] nrm_x, double[:, ::1] nrm_y, double[:, ::1] diff2_x, double[:, ::1] diff2_y, double[:, ::1] inv_sum_sd2s_x, double[:, ::1] inv_sum_sd2s_y, double[::1] v_l0_exp_hist):
    #  Value of pdf @ fc.  
    #  fxd_IIQ2    1./q2_c
    #  Mc          - spiking + prior  mean
    #  Sigma2c     - spiking + prior  variance
    #  p_riemann_x - points at which integral discretized
    #  p_px        - occupancy
    cdef double tmp
    cdef double *p_l0_exp_hist = &v_l0_exp_hist[0]
    cdef int mi, i, m
    cdef double* p_ap_Ns  = &ap_Ns[0]
    cdef double* p_nrm_x  = &nrm_x[0, 0]
    cdef double* p_nrm_y  = &nrm_y[0, 0]
    cdef double* p_diff2_x = &diff2_x[0, 0]
    cdef double* p_diff2_y = &diff2_y[0, 0]
    cdef double* p_inv_sum_sd2s_x = &inv_sum_sd2s_x[0, 0]
    cdef double* p_inv_sum_sd2s_y = &inv_sum_sd2s_y[0, 0]

    with nogil:
        for m in xrange(M):
            tmp = 0
            for i in xrange(totalpcs):
                mi  = m*totalpcs + i
                tmp += p_ap_Ns[i]*p_nrm_x[mi]*p_nrm_y[mi]*exp(-0.5*p_diff2_x[mi]*p_inv_sum_sd2s_x[mi] - 0.5*p_diff2_y[mi]*p_inv_sum_sd2s_y[mi])
            tmp *= dt
            p_l0_exp_hist[m] = tmp
    

###########################################################################
###########################################################################
###########################################################################
#####################  OCCUPATION FUNCTIONS

@cython.cdivision(True)
def nrm_xy(int totalpcs, int M, double[:, ::1] inv_sum_sd2s, double[:, ::1] nrm, double[::1] q2s, double[::1] ap_sd2s):
    """nn
    nrm_y   
    ONLY CALLED ONCE, .  SAME IS CALCULATED and integrated in smp_q2
    """
    cdef double* p_nrms = &nrm[0, 0]
    cdef double* p_inv_sum_sd2s = &inv_sum_sd2s[0, 0]
    cdef double* p_q2s = &q2s[0]
    cdef double* p_ap_sd2s = &ap_sd2s[0]
    cdef double tmp
    cdef int m, i

    with nogil:
        for m in xrange(M):
            #printf("p_q2s[%d] %.3e\n", m, p_q2s[m])
            for i in xrange(totalpcs):
                tmp = 1./(p_q2s[m] + p_ap_sd2s[i])
                p_inv_sum_sd2s[m*totalpcs + i] = tmp
                #printf("%d %d   %.3e %.3e\n", m, i, p_q2s[m], p_ap_sd2s[i])
                p_nrms[m*totalpcs + i] = isqrt_twpi * sqrt(tmp)

def diff2_xy(int totalpcs, int M, double[:, ::1] diffs2, double[::1] fs, double[::1] ap_mn):
    """
    squared displacement in 1 dimension between cluster center and path 
    ONLY CALLED ONCE, .  SAME IS CALCULATED and integrated in smp_f
    """
    cdef double* p_diffs2 = &diffs2[0, 0]
    cdef double* p_fs = &fs[0]
    cdef double* p_ap_mn = &ap_mn[0]
    cdef double tmp
    cdef int m, i

    with nogil:
        for m in xrange(M):
            for i in xrange(totalpcs):
                p_diffs2[m*totalpcs + i] = (p_fs[m]-p_ap_mn[i])*(p_fs[m]-p_ap_mn[i])



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def smp_f(int cmpnt, int itr, int M, double[::1] x_or_yt0t1, long[::1] clstsz, long[::1] cls_strt_inds, 
          long[::1] sts, int t0, double[::1] l0, 
          int totalpcs, double[::1] ap_Ns, double[::1] mns, 
          double[:, ::1] diff1, double[:, ::1] nrm1, double[:, ::1] inv_sum_sd2s1,
          double[:, ::1] diff2, double[:, ::1] nrm2, double[:, ::1] inv_sum_sd2s2,
          double[::1] _f_u, double[::1] _f_q2, double[::1] m_rands, 
          double[::1] f_out, double[::1] q2, int ignore_space):
    """
    f_smv is the component of being sampled, f_fx is the fixed component
    _f_sm_u are the prior hyperparameters for f_xmv
    """
    global f_STEPS, f_SMALL, f_cldz, fx2, cpf2
    global f_smpld_support
    global adtv_pdf_params
    global __NRM
    global dt
    """
    f     parameter f
    _f_u  
    """
    cdef int m, i
    cdef double tmp
    cdef double[::1]     v_adtv_pdf_params = adtv_pdf_params
    cdef double* p_adtv_pdf_params = &v_adtv_pdf_params[0]

    cdef double* _p_q2pr       #  prior hyperparameter
    cdef double* _p_f_u        #  prior hyperparameter
    cdef double* p_f           #  component of f to be sampled
    cdef double* p_l0     = &l0[0]    
    cdef double* p_q2          #  q2 of that component
    cdef double fs, fq2        #  temporary variables

    cdef double* p_pt0t1

    _p_q2pr = &_f_q2[0]      #  prior hyperparameter
    _p_f_u  = &_f_u[0]       #  prior hyperparameter
    p_pt0t1 = &x_or_yt0t1[0]
    p_f     = &f_out[0]
    p_q2    = &q2[0]
    cdef long* p_clstsz  = &clstsz[0]
    cdef long* p_strt_inds = &cls_strt_inds[0]
    cdef long* p_sts     = &sts[0]
    cdef double U, FQ2
    cdef int mtotalpcs

    cdef double *p_diff1 = &diff1[0, 0]   #  M_use x totalpcs
    cdef double *p_nrm1  = &nrm1[0, 0]    #  M_use x totalpcs
    cdef double *p_inv_sum_sd2s1  = &inv_sum_sd2s1[0, 0]  #  M_use x totalpcs
    cdef double *p_diff2 = &diff2[0, 0]   #  M_use x totalpcs
    cdef double *p_nrm2  = &nrm2[0, 0]    #  M_use x totalpcs
    cdef double *p_inv_sum_sd2s2  = &inv_sum_sd2s2[0, 0]  #  M_use x totalpcs
    cdef double *p_ap_Ns = &ap_Ns[0]

    cdef double *p_mns  = &mns[0]
    cdef double *p_m_rnds = &m_rands[0]
    global p_fx2, p_cpf2

    with nogil:
        for 0 <= m < M:
            mtotalpcs = m*totalpcs
            if p_clstsz[m] > 0:
                tmp = 0
                for p_strt_inds[m] <= i < p_strt_inds[m+1]:
                    tmp += p_pt0t1[p_sts[i]-t0]

                fs = tmp/p_clstsz[m]      #  spiking part
                fq2= p_q2[m]/p_clstsz[m]
                U = (fs*_p_q2pr[m] + _p_f_u[m]*fq2) / (_p_q2pr[m] + fq2)
                FQ2 = (_p_q2pr[m]*fq2) / (_p_q2pr[m] + fq2)
            else:
                U   = _p_f_u[m]
                FQ2 = _p_q2pr[m]

            #print("****   %(1)d   %(2)d       %(3)d   %(4)d" % {"1" : p_pt0t1[p_sts[p_strt_inds[m]-t0]], "2" : p_pt0t1[p_sts[p_strt_inds[m+1]-t0]], "3" : p_sts[p_strt_inds[m]-t0], "4" : p_sts[p_strt_inds[m+1]-t0]})

            #print("clstsz %(m)d    %(tmp).3f  %(fq2).3e  fs %(fs).3e  U  %(U).3f   FQ2  %(FQ2).3e" % {"U" : U, "FQ2" : FQ2, "m" : clstsz[m], "fq2" : fq2, "fs" : fs, "tmp" : tmp})
            p_adtv_pdf_params[0] = U
            p_adtv_pdf_params[1] = FQ2
            
            #  stay same:  diff1, nrm1, inv_sum_sd2s1
            #  changes:    diff2, nrm2, inv_sum_sd2s2
            p_f[m] = adtv_support_pdf(totalpcs, p_ap_Ns, p_fx2, p_cpf2, p_f_smpld_support, f_STEPS, f_cldz, f_SMALL, dt, p_l0[m], __NRM, p_adtv_pdf_params, p_mns, &(p_diff1[mtotalpcs]), &(p_nrm1[mtotalpcs]), &(p_inv_sum_sd2s1[mtotalpcs]), &(p_diff2[mtotalpcs]), &(p_nrm2[mtotalpcs]), &(p_inv_sum_sd2s2[mtotalpcs]), m, p_m_rnds[m])

            for 0 <= i < totalpcs:
                p_diff2[mtotalpcs + i] = (p_f[m]-p_mns[i])*(p_f[m]-p_mns[i])


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def smp_q2(int cmpnt, int itr, int M, double[::1] x_or_yt0t1, long[::1] clstsz, long[::1] cls_strt_inds, 
           long[::1] sts, int t0, double[::1] l0, 
           int totalpcs, double[::1] ap_Ns, double[::1] sds, 
           double[:, ::1] diff1, double[:, ::1] nrm1, double[:, ::1] inv_sum_sd2s1,
           double[:, ::1] diff2, double[:, ::1] nrm2, double[:, ::1] inv_sum_sd2s2,
           double[::1] _q2_a, double[::1] _q2_B, double[::1] m_rands, 
           double[::1] f, double[::1] q2_out, int ignore_space):
    """
    f_smv is the component of being sampled, f_fx is the fixed component
    _f_sm_u are the prior hyperparameters for f_xmv
    """
    global q2_STEPS, q2_SMALL, q2_cldz, fx2, cpf2
    global q2_smpld_support
    global adtv_pdf_params
    global __IG
    global dt
    global isqrt_twpi
    """
    f     parameter f
    _f_u  
    """
    cdef int m, i
    cdef double tmp
    cdef double[::1]     v_adtv_pdf_params = adtv_pdf_params
    cdef double* p_adtv_pdf_params = &v_adtv_pdf_params[0]

    cdef double* _p_q2_a = &_q2_a[0]
    cdef double* _p_q2_B  = &_q2_B[0]
    cdef double* p_f           #  component of f to be sampled
    cdef double* p_l0     = &l0[0]        
    cdef double* p_q2          #  q2 of that component
    cdef double fs, fq2        #  temporary variables

    cdef double* p_pt0t1 
    _p_q2_a = &_q2_a[0]
    _p_q2_B = &_q2_B[0]
    p_pt0t1 = &x_or_yt0t1[0]
    p_f     = &f[0]
    p_q2    = &q2_out[0]

    cdef long* p_clstsz  = &clstsz[0]
    cdef long* p_strt_inds = &cls_strt_inds[0]
    cdef long* p_sts     = &sts[0]
    cdef double SL_a, SL_B

    cdef double *p_diff1 = &diff1[0, 0]   #  M_use x totalpcs
    cdef double *p_nrm1  = &nrm1[0, 0]    #  M_use x totalpcs
    cdef double *p_inv_sum_sd2s1  = &inv_sum_sd2s1[0, 0]  #  M_use x totalpcs
    cdef double *p_diff2 = &diff2[0, 0]   #  M_use x totalpcs
    cdef double *p_nrm2  = &nrm2[0, 0]    #  M_use x totalpcs
    cdef double *p_inv_sum_sd2s2  = &inv_sum_sd2s2[0, 0]  #  M_use x totalpcs
    
    cdef double *p_ap_Ns = &ap_Ns[0]

    cdef double *p_sds  = &sds[0]
    cdef double *p_m_rnds = &m_rands[0]
    cdef int mtotalpcs

    global p_qx2, p_cpq22        

    #  v_sts   spike times, (5 10 11 16) (3 7 9)
    with nogil:
        for m in xrange(M):
            mtotalpcs = m*totalpcs
            if p_clstsz[m] > 0:
                SL_B= 0
                for p_strt_inds[m] <= i < p_strt_inds[m+1]:
                    SL_B += (p_pt0t1[p_sts[i]-t0]-p_f[m])*(p_pt0t1[p_sts[i]-t0]-p_f[m])
                SL_B *= 0.5
                SL_B += _p_q2_B[m]

                #  -S/2 (likelihood)  -(a+1)
                SL_a = 0.5*p_clstsz[m] + _p_q2_a[m] + 1
            else:
                SL_a = _p_q2_a[m]
                SL_B = _p_q2_B[m]
            #if (itr > 4500) and (itr % 100 == 0) and (m == 0):
            #    printf("SL_a   %.3f  SL_B   %.3f\n", SL_a, SL_B)

            p_adtv_pdf_params[0] = SL_a
            p_adtv_pdf_params[1] = SL_B

            #  stay same:  diff1, diff2, nrm1, inv_sum_sd2s1
            #  changes:    nrm2, inv_sum_sd2s2
            p_q2[m] = adtv_support_pdf(totalpcs, p_ap_Ns, p_qx2, p_cpq22, p_q2_smpld_support, q2_STEPS, q2_cldz, f_SMALL, dt, p_l0[m], __IG, p_adtv_pdf_params, p_sds, &(p_diff1[mtotalpcs]), &(p_nrm1[mtotalpcs]), &(p_inv_sum_sd2s1[mtotalpcs]), &(p_diff2[mtotalpcs]), &(p_nrm2[mtotalpcs]), &(p_inv_sum_sd2s2[mtotalpcs]), m, p_m_rnds[m])

            for 0 <= i < totalpcs:
                tmp = 1./(p_q2[m] + p_sds[i])
                p_inv_sum_sd2s2[mtotalpcs + i] = tmp
                p_nrm2[mtotalpcs + i] = isqrt_twpi * sqrt(tmp)

            # if itr % 10000 == 0:
            #     dat = _N.empty((len(adtvInds), 2))
            #     dat[:, 0] = qx2[adtvInds]
            #     dat[:, 1] = cpq22[adtvInds]
            #     printf("!!!!!!!!!! %.4e %.4e\n", SL_a, SL_B)
            #     fn = _U.uniqFN("adtv%d" % m, serial=True, iStart=0)
            #     _N.savetxt(fn, dat, fmt="%.4e %.4e")

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# def adtv_support_pdf(double[::1] gx, double[::1] cond_pstr,
#                      unsigned char[::1] smpld_support,
#                      int STEPS, int cldz, int small,
#                      double dt, double l0, 
#                      int dist, double[::1] params,
#                      int m, double[::1] m_rnds):

cdef double adtv_support_pdf(int totalpcs, double *p_ap_Ns,
                     double* p_gx, double *p_cond_pstr,
                     unsigned char *p_smpld_support,
                     int STEPS, int cldz, int small,
                     double dt, double l0, 
                     int dist, double *p_params, double* p_mns_o_sds,
                     double *p_diff1, 
                     double *p_nrm1, double *p_inv_sum_sd2s1,
                     double *p_diff2, 
                     double *p_nrm2, double *p_inv_sum_sd2s2,
                     int m, double m_rnd) nogil:

    """
    This gets called M times per Gibbs sample
    gx      grid @ which to sample pdf.  
    cond_pstr      storage of sampled values
    STEPS   gx has size 2**STEPS+1
    cldz    current level of discretization
    small   small probability
    dist    __NRM or __IT
    params  
    rieman_x  for spatial integral, grid for integration
    px      occupation
    """

    #  STEPS= 5,   IS = 3
    #  cldz=3, skp=2**cldz, start=0    stop=2**STEPS+1
    #  0 8 16 24 32           32/8 + 1  = 5 elements
    #  skp/2  = 4
    #
    #  cldz=2, skp=2**(cldz+1), start=2**cl 
    #   4 12 20 28              # 4 elements    2**2  
    #  skp/4 = 2    next lvl
    #
    #  cldz=1, skp=2**(cldz+1)
    #  2 6 10 14 18 22 26 30    # 8 elements     skp/4 = 1
    #
    #  cldz=0, skp=2**(cldz+1)
    #  1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31   # 16 elements

    #cldz = 5   # current level of discretization.  initl_skp  = 2**cldz
    #  Skip sizes go like 8 8 4 2.   2**

    global minSmps, p_adtvInds
    

    cdef int bDone= 0

    cdef double pmax = -1.e300
    cdef double gpmax= -1.e300   #  global pmax

    cdef int lvl  = 0  #  how deep in discretization level.  maximum val of cldz+1
    cdef int L

    cdef int iBdL, iBdR
    cdef int iBdL_glbl = p_twoPOW[STEPS]+2#2**STEPS+2
    cdef int iBdR_glbl = -1

    cdef int nSmpd      = 0
    cdef int ix, skp, strt, stop, nxt_lvl_skp, initl_skp
    cdef double pthresh

    #cdef long* p_adtvInds = &v_adtvInds[0]
    #cdef double *p_cond_pstr = &cond_pstr[0]
    #cdef double *p_gx        = &gx[0]     #  grid on which post(param) sampled
    #cdef unsigned char *p_smpld_support = &smpld_support[0]

    #cdef double *p_params    = &params[0]

    # cdef double *p_diff1 = &diff1[0, 0]
    # cdef double *p_nrm1  = &nrm1[0, 0]
    # cdef double *p_inv_sum_sd2s1  = &inv_sum_sd2s1[0, 0]
    # cdef double *p_nrm2  = &nrm2[0, 0]
    # cdef double *p_inv_sum_sd2s2  = &inv_sum_sd2s2[0, 0]

    cdef int imax   = -1
    cdef int gimax  = -1

    cdef int n_smpld_pts  = 0

    ####  
    ####  params conditioned on
    ####  priors    
    cdef double _q2_a, _q2_B
    cdef double _f_u, _f_q2
    #_q2_a, _q2_B
    #_f_u, _f_q2
    ####  likelihood terms :  f_c
    cdef double fc = 0
    cdef long Nupx, Nupy, iStart_xy, iStart_x, iStart_y
    cdef double ibnsz_x, ibnsz_y
    cdef int istepbck = 0
    cdef double t1, t2, t3
    cdef int iAdtvInds, ii
    cdef double retRnd, rnd

    cdef double Mc, Sigma2c, q2_cnd_on, a, B, f_cnd_on, iq2_cnd_on, f_smp_cmp, q2_smp_cmp

    if dist == __NRM:
        Mc        = p_params[0]   #  observed spikes part
        Sigma2c   = p_params[1]   #  observed spikes part
    else:
        a         = p_params[0]   #  observed spikes part
        B         = p_params[1]   #  observed spikes part

    #initial point, set pmax, gmax
    if dist == __NRM:   ################## __NRM
        p_cond_pstr[0] = pdfNRM(totalpcs, p_ap_Ns, dt, l0, Mc, Sigma2c, p_gx[0], p_mns_o_sds, p_diff1, p_nrm1, p_inv_sum_sd2s1, p_nrm2, p_inv_sum_sd2s2)
        #p_cond_pstr[0] = pdfNRM(totalpcs, dt, l0, Mc, Sigma2c, p_mns_o_sds, p_diff1, p_nrm1, p_inv_sum_sd2s1, p_diff2, p_nrm2, p_inv_sum_sd2s2)
    if dist == __IG:   ################## __NRM
        p_cond_pstr[0] = pdfIG(totalpcs, p_ap_Ns, dt, l0, a, B, p_gx[0], p_mns_o_sds, p_diff1, p_nrm1, p_inv_sum_sd2s1, p_diff2)

    pmax = p_cond_pstr[0]
    gmax = pmax
    imax = 0

    while (cldz > -1) and (bDone == 0):
        def_at_skp = p_twoPOW[cldz]
        if lvl == 0:  # 1st 2 passes skip sizes are the same
            strt = 0   #  first point will have been done already
            skp  = p_twoPOW[cldz]          # 2**cldz
            #print "init skip %d skp" % skp
            initl_skp  = skp
            stop  = p_twoPOW[STEPS]-strt   # 2**STEPS-strt
            nxt_lvl_skp = skp/2
        else:
            strt = iBdL - nxt_lvl_skp
            if strt < 0:
                strt = iBdL + nxt_lvl_skp
            stop = iBdR + nxt_lvl_skp
            if stop > p_twoPOW[STEPS]:      # 2**STEPS:
                stop = iBdR - nxt_lvl_skp
            skp  = p_twoPOW[cldz+1]        # 2**(cldz+1)
            nxt_lvl_skp = skp/4

        ################  these 2 must be done together, and after strt, skp set
        cldz  -= 1
        lvl   += 1
        ################  these 2 must be done together, and after strt, skp set
        if dist == __NRM:
            ##  for each f cond posterior calculated at:
            #   - q2 conditioned on
            #   - Mc (likelihood + prior) center of observation
            #   - Sigma2c (likelihood + prior) width of observation
            #   prior

            for strt <= ix < stop+1 by skp:
                p_cond_pstr[ix] = pdfNRM(totalpcs, p_ap_Ns, dt, l0, Mc, Sigma2c, p_gx[ix], p_mns_o_sds, p_diff1, p_nrm1, p_inv_sum_sd2s1, p_nrm2, p_inv_sum_sd2s2)

                n_smpld_pts += 1
                p_smpld_support[ix] = 1
                if p_cond_pstr[ix] > pmax:
                    pmax = p_cond_pstr[ix]   # pmax updated each time grid made finer
                    imax = ix
        if dist == __IG:
            for strt <= ix < stop+1 by skp:
                p_cond_pstr[ix] = pdfIG(totalpcs, p_ap_Ns, dt, l0, a, B, p_gx[ix], p_mns_o_sds, p_diff1, p_nrm1, p_inv_sum_sd2s1, p_diff2)
                n_smpld_pts += 1
                p_smpld_support[ix] = 1                

                if p_cond_pstr[ix] > pmax:
                    pmax = p_cond_pstr[ix]   # pmax updated each time grid made finer
                    imax = ix

        if pmax > gpmax:   #  stop when 
            gpmax = pmax
            pthresh= gpmax-small   #  _N.exp(-12) == 6.144e-6
            gimax = imax

        #  start half the current skip size before and after iBdL and iBdR
        # now find left and right bounds

        ix = strt  #  sampling has started from istrt.  

        #printf("strt %d     stop %d     skip %d\n", strt, stop, skp)
        istepbck = 0
        while (ix < gimax) and (p_cond_pstr[ix] < pthresh):
            # if we don't do this while, we still end up stepping back. 
            istepbck += 1
            ix += skp  
        #  at this point, p_cond_str[ix] >= pthresh or ix >= gimax
        if istepbck > 0:
            ix -= skp  # step back, so p_cond_str[ix] is now < pthresh.
        if ix >= 0:
            iBdL = ix
        else:
            while ix < 0:
                ix += skp
            iBdL = ix
        ##########
        istepbck = 0
        ix = stop
        while (ix > gimax) and (p_cond_pstr[ix] < pthresh):
            istepbck += 1
            ix -= skp
        if istepbck > 0:
            ix += skp

        if ix <= p_twoPOW[STEPS]:
            iBdR = ix
        else:     #  rewind if we go past right limit.  Happens when distribution too wide, right hand is still > pthresh
            while ix > p_twoPOW[STEPS]:
                ix -= skp
            iBdR = ix

        iBdL_glbl = iBdL if (iBdL < iBdL_glbl) else iBdL_glbl
        iBdR_glbl = iBdR if (iBdR > iBdR_glbl) else iBdR_glbl

        nSmpd += (iBdR - iBdL) / skp + 1

        bDone = 1 if (nSmpd > minSmps) else 0

    iAdtvInds = 0
    #for ii in range(0, p_twoPOW[STEPS]+1, def_at_skp):
    for 0 <= ii < p_twoPOW[STEPS]+1 by def_at_skp:#in range(0, p_twoPOW[STEPS]+1, def_at_skp):
        if p_smpld_support[ii] == 1:
            p_cond_pstr[ii] -= gpmax
            p_adtvInds[iAdtvInds] = ii
            iAdtvInds += 1
            p_smpld_support[ii] = 0
    #printf("ccccccccccccccccccccc  %d   %d\n", iAdtvInds, n_smpld_pts)
    p_p[0]  = exp(p_cond_pstr[p_adtvInds[0]])

    for 1 <= ii < n_smpld_pts:
        p_p[ii] = exp(p_cond_pstr[p_adtvInds[ii]])
        p_dx[ii-1] = p_gx[p_adtvInds[ii]]-p_gx[p_adtvInds[ii-1]]
        p_cdf[ii] = p_cdf[ii-1] + 0.5*(p_p[ii-1]+p_p[ii])*p_dx[ii-1]#*itot
    for 1 <= ii < n_smpld_pts:
        p_cdf[ii] /= p_cdf[n_smpld_pts-1]     #  even if U[0,1] rand is 1, we still have some room at the end to add a bit of noise.

    #for 1 <= ii < n_smpld_pts:            
        if (m_rnd >= p_cdf[ii-1]) and (m_rnd <= p_cdf[ii]):
            break
    retRnd = p_gx[p_adtvInds[ii-1]] + ((m_rnd - p_cdf[ii-1]) / (p_cdf[ii] - p_cdf[ii-1])) * p_dx[ii-1]  # unlike in above case, retRnd may be < 0            
    # printf("***  m %d  ", m)
    # printf("retRnd %f  ", retRnd)
    # printf("ii %d\n", ii)
    
    return retRnd#adtvInds, n_smpld_pts
        

########################################################################
@cython.cdivision(True)
#  fc, q2
cdef double pdfNRM(int totalpcs, double *p_ap_Ns, double dt, double l0, double Mc, double Sigma2c, double f_smpld, double *p_path_mn, double *p_diffs1, double *p_nrm1, double *p_inv_sum_sd2s1, double *p_nrm2, double *p_inv_sum_sd2s2) nogil: 
    #  Value of pdf @ fc.  
    #  fxd_IIQ2    1./q2_c
    #  Mc          - spiking + prior  mean
    #  Sigma2c     - spiking + prior  variance
    #  p_riemann_x - points at which integral discretized
    #  p_px        - occupancy

    #  diffs1, nrm1, inv_sum_sd2s1   --  fixed component
    #          nrm2, inv_sum_sd2s2   --  component where f_{xy} is sampled (fc)

    cdef double sptlIntgrl = 0
    cdef int pcs
    
    for 0 <= pcs < totalpcs:
        #printf("%d %.3e %.3e %.3e   %.3e %.3e %.3e\n", pcs, p_nrm1[pcs], p_nrm2[pcs], p_inv_sum_sd2s1[pcs], p_inv_sum_sd2s2[pcs], f_smpld, p_path_mn[pcs])
        sptlIntgrl += p_ap_Ns[pcs]*p_nrm1[pcs]*p_nrm2[pcs]*exp(-0.5*p_diffs1[pcs]*p_inv_sum_sd2s1[pcs] - 0.5*(f_smpld - p_path_mn[pcs])*(f_smpld - p_path_mn[pcs]) * p_inv_sum_sd2s2[pcs])
    sptlIntgrl *= dt*l0

    return -0.5*(f_smpld-Mc)*(f_smpld-Mc)/Sigma2c-sptlIntgrl


@cython.cdivision(True)
#  fc, q2
cdef double pdfIG(int totalpcs, double *p_ap_Ns, double dt, double l0, double a, double B, double q2_smpld, double *p_path_sd2, double *p_diffs1, double *p_nrm1, double *p_inv_sum_sd2s1, double *p_diffs2) nogil: 
    #  Value of pdf @ fc.  
    #  fxd_IIQ2    1./q2_c
    #  Mc          - spiking + prior  mean
    #  Sigma2c     - spiking + prior  variance
    #  p_riemann_x - points at which integral discretized
    #  p_px        - occupancy

    #  diffs1, nrm1, inv_sum_sd2s1   --  fixed component
    #          nrm2, inv_sum_sd2s2   --  component where f_{xy} is sampled (fc)

    cdef double sptlIntgrl = 0
    cdef int pcs
    cdef double sum_vars = 0
    
    for 0 <= pcs < totalpcs:
        sum_vars = q2_smpld + p_path_sd2[pcs]
        sptlIntgrl += (p_ap_Ns[pcs]*p_nrm1[pcs]) / sqrt(twpi*sum_vars)*exp(-0.5*p_diffs1[pcs]*p_inv_sum_sd2s1[pcs] - 0.5*p_diffs2[pcs] / sum_vars)
    sptlIntgrl *= dt*l0

    return -(a + 1)*log(q2_smpld) - B/q2_smpld-sptlIntgrl
