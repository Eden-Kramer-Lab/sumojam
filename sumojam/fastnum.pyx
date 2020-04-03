#  do multiple quadratics
cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free
from libc.stdio cimport printf
from libc.math cimport exp
import numpy as _N
cimport numpy as _N

                
#@cython.boundscheck(False)
#@cython.wraparound(False)
def cluster_probabilities(double[::1] pkFR, double [::1] mkNrms, double[:, ::1] exp_arg, double[:, ::1] econt, double[:, ::1] rat, double[::1] rnds, double[::1] f, double [::1] xAS, double[::1] iq2, double [:, ::1] qdrSPC, double[:, ::1] mAS, double [:, ::1] u, double[:, :, ::1] iSg, double[:, ::1] qdrMKS, double[::1] v_max_so_far, int M, int N, int k):

                  
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int n, m, mm, mNk, mkk, nk, ik, mNk_nk, mN, mNn, mK, nK, mNpn

    cdef double *p_mAS   = &mAS[0, 0]
    cdef double *p_u     = &u[0, 0]
    #cdef double *p_v   = &v[0, 0, 0]
    cdef double *p_iSg = &iSg[0, 0, 0]
    #cdef double *p_qdrMKS = &qdrMKS[0, 0]
    cdef double s_qdrMKS, s_qdrSPC

    cdef double pfm
    cdef double piq2m
    #cdef double *p_qdrSPC   = &qdrSPC[0, 0]
    #cdef double *p_fr       = &fr[0, 0]
    cdef double *p_f       = &f[0]
    #cdef double *p_xASr     = &xASr[0, 0]
    #cdef double *p_iq2r     = &iq2r[0, 0]
    cdef double *p_xAS     = &xAS[0]
    cdef double *p_iq2     = &iq2[0]
    cdef double *p_max_so_far = &v_max_so_far[0]

    cdef double *p_pkFR       = &pkFR[0]
    cdef double *p_mkNrms     = &mkNrms[0]
    cdef double *p_exp_arg       = &exp_arg[0, 0]
    cdef double *p_econt         = &econt[0, 0]
    cdef double *p_rat           = &rat[0, 0]
    cdef double *p_rnds         = &rnds[0]        
    cdef double pkFRr_m, mkNrms_m

    cdef double iS00
    cdef double iS11
    cdef double iS22
    cdef double iS33
    cdef double iS01
    cdef double iS02
    cdef double iS03
    cdef double iS12
    cdef double iS13
    cdef double iS23
    cdef double d0
    cdef double d1
    cdef double d2
    cdef double d3
    # cdef double u_m_0
    # cdef double u_m_1
    # cdef double u_m_2
    # cdef double u_m_3


    #with nogil:
    for 0 <= m < M-1:
        mkk  = m*k*k
        mN   = m*N
        #mNk   = m*N*k
        #mNk   = m*N*k
        mK =   m*k

        #  write output to p_qdr[m, n].  (xn - um)iSg_m (xn - um)

        iS00 = p_iSg[mkk]
        iS11 = p_iSg[mkk+k+1]
        iS22 = p_iSg[mkk+2*k+2]
        iS33 = p_iSg[mkk+3*k+3]
        iS01 = 2*p_iSg[mkk+1]   #  this
        iS02 = 2*p_iSg[mkk+2]
        iS03 = 2*p_iSg[mkk+3]
        iS12 = 2*p_iSg[mkk+k+2]
        iS13 = 2*p_iSg[mkk+k+3]
        iS23 = 2*p_iSg[mkk+2*k+3]

        u_m_0 = p_u[mK]
        u_m_1 = p_u[mK + 1]
        u_m_2 = p_u[mK + 2]
        u_m_3 = p_u[mK + 3]


        pfm = p_f[m]
        piq2m= p_iq2[m]

        pkFR_m = p_pkFR[m]
        mkNrms_m = p_mkNrms[m]

        for n in range(N):
            #mNk_nk = mNk + n*k
            mNn    = mN+n
            nK     = n*k

            d0 = p_mAS[nK]-p_u[mK]
            d1 = p_mAS[nK+1]-p_u[mK+1]
            d2 = p_mAS[nK+2]-p_u[mK+2]
            d3 = p_mAS[nK+3]-p_u[mK+3]               
            #                p_qdrMKS[mNn] = d0*(iS00*d0 + iS01*d1 + iS02*d2 + iS03*d3)+\
            s_qdrMKS = d0*(iS00*d0 + iS01*d1 + iS02*d2 + iS03*d3)+\
                d1*(iS11*d1 + iS12*d2 + iS13*d3) +\
                d2*(iS22*d2 + iS23*d3) +\
                d3*iS33*d3

            #p_qdrSPC[mNn] = (pfm - p_xAS[n])*(pfm - p_xAS[n])*piq2m
            s_qdrSPC = (pfm - p_xAS[n])*(pfm - p_xAS[n])*piq2m

            #p_exp_arg[mNn] = pkFR_m + mkNrms_m - 0.5*(p_qdrSPC[mNn] + p_qdrMKS[mNn])
            s_exp_arg      = pkFR_m + mkNrms_m - 0.5*(s_qdrSPC + s_qdrMKS)
            if s_exp_arg > p_max_so_far[n]:
                p_max_so_far[n] = s_exp_arg
            p_exp_arg[mNn] = s_exp_arg


    m    = M-1
    mkk  = m*k*k
    mN   = m*N
    #mNk   = m*N*k
    #mNk   = m*N*k
    mK =   m*k

    #  write output to p_qdr[m, n].  (xn - um)iSg_m (xn - um)

    iS00 = p_iSg[mkk]
    iS11 = p_iSg[mkk+k+1]
    iS22 = p_iSg[mkk+2*k+2]
    iS33 = p_iSg[mkk+3*k+3]
    iS01 = 2*p_iSg[mkk+1]   #  this
    iS02 = 2*p_iSg[mkk+2]
    iS03 = 2*p_iSg[mkk+3]
    iS12 = 2*p_iSg[mkk+k+2]
    iS13 = 2*p_iSg[mkk+k+3]
    iS23 = 2*p_iSg[mkk+2*k+3]

    u_m_0 = p_u[mK]
    u_m_1 = p_u[mK + 1]
    u_m_2 = p_u[mK + 2]
    u_m_3 = p_u[mK + 3]


    pfm = p_f[m]
    piq2m= p_iq2[m]

    pkFR_m = p_pkFR[m]
    mkNrms_m = p_mkNrms[m]

    for n in range(N):
        #mNk_nk = mNk + n*k
        mNn    = mN+n
        nK     = n*k

        d0 = p_mAS[nK]-p_u[mK]
        d1 = p_mAS[nK+1]-p_u[mK+1]
        d2 = p_mAS[nK+2]-p_u[mK+2]
        d3 = p_mAS[nK+3]-p_u[mK+3]               
        #p_qdrMKS[mNn] = d0*(iS00*d0 + iS01*d1 + iS02*d2 + iS03*d3)+\
        s_qdrMKS = d0*(iS00*d0 + iS01*d1 + iS02*d2 + iS03*d3)+\
            d1*(iS11*d1 + iS12*d2 + iS13*d3) +\
            d2*(iS22*d2 + iS23*d3) +\
            d3*iS33*d3

        #p_qdrSPC[mNn] = (pfm - p_xAS[n])*(pfm - p_xAS[n])*piq2m
        s_qdrSPC = (pfm - p_xAS[n])*(pfm - p_xAS[n])*piq2m

        #p_exp_arg[mNn] = pkFR_m + mkNrms_m - 0.5*(p_qdrSPC[mNn] + p_qdrMKS[mNn])
        s_exp_arg = pkFR_m + mkNrms_m - 0.5*(s_qdrSPC + s_qdrMKS)

        if s_exp_arg > p_max_so_far[n]:
            p_max_so_far[n] = s_exp_arg
        p_exp_arg[mNn] = s_exp_arg                

        for 0 <= mm < M:
            p_econt[mm*N+n] = exp(p_exp_arg[mm*N+n]-p_max_so_far[n])



            
    for 0 <= m < M:
        mN = m*N
        for 0 <= n < N:
            mNpn = mN+n
            #  [m+1, n]  = [m, n] + [m, n]
            #  (m+1)*N + n = mN + n + N
            #p_rat[mNpn + N] = p_rat[mNpn] + p_econt[mNpn]
            #p_rat[(m+1)*N + n] = p_rat[mNpn] + p_econt[mNpn]
            rat[m+1, n] = rat[m, n] + econt[m, n]

    for 0 <= n < N:   #  rat[M, n] = M*N + n
        #  rnds[0] *= rat[M, 0]
        #     p_rnds[n] *= p_rat[M*N + 0]
        #  rnds[1] *= rat[M, 1]
        #     p_rnds[n] *= p_rat[M*N + 1]            
        #p_rnds[n] *= p_rat[M*N+n]
        rnds[n] *= rat[M, n]
                

    # printf("%.4e\n", p_rat[M*N])
    # print("%.4e" % rat[M, 0])
    # printf("%.4e\n", p_rat[M*N+5])
    # print("%.4e" % rat[M, 5])    
    # printf("%.4e\n", p_rat[M*N+N-1])
    # print("%.4e" % rat[M, N-1])    



#@cython.boundscheck(False)
#@cython.wraparound(False)
def cluster_probabilities_2d(double[::1] pkFR, double [::1] mkNrms, double[:, ::1] exp_arg, double[:, ::1] econt, double[:, ::1] rat, double[::1] rnds, double[::1] fx, double[::1] fy, double [::1] xAS, double [::1] yAS, double[::1] iq2x, double[::1] iq2y, double [:, ::1] qdrSPC, double[:, ::1] mAS, double [:, ::1] u, double[:, :, ::1] iSg, double[:, ::1] qdrMKS, double[::1] v_max_so_far, int M, int N, int k):

                  
    #  fxs       M x fss   
    #  fxrux     Nupx    
    #  f_intgrd  Nupx
    cdef int n, m, mm, mNk, mkk, nk, ik, mNk_nk, mN, mNn, mK, nK, mNpn

    cdef double *p_mAS   = &mAS[0, 0]
    cdef double *p_u     = &u[0, 0]
    #cdef double *p_v   = &v[0, 0, 0]
    cdef double *p_iSg = &iSg[0, 0, 0]
    #cdef double *p_qdrMKS = &qdrMKS[0, 0]
    cdef double s_qdrMKS, s_qdrSPC

    cdef double pfxm, pfym
    cdef double piqx2m, piqy2m
    #cdef double *p_qdrSPC   = &qdrSPC[0, 0]
    #cdef double *p_fr       = &fr[0, 0]
    cdef double *p_fx       = &fx[0]
    cdef double *p_fy       = &fy[0]    
    #cdef double *p_xASr     = &xASr[0, 0]
    #cdef double *p_iq2r     = &iq2r[0, 0]
    cdef double *p_xAS     = &xAS[0]
    cdef double *p_yAS     = &yAS[0]    
    cdef double *p_iqx2     = &iq2x[0]
    cdef double *p_iqy2     = &iq2y[0]    
    cdef double *p_max_so_far = &v_max_so_far[0]

    cdef double *p_pkFR       = &pkFR[0]
    cdef double *p_mkNrms     = &mkNrms[0]
    cdef double *p_exp_arg       = &exp_arg[0, 0]
    cdef double *p_econt         = &econt[0, 0]
    cdef double *p_rat           = &rat[0, 0]
    cdef double *p_rnds         = &rnds[0]        
    cdef double pkFRr_m, mkNrms_m

    cdef double iS00
    cdef double iS11
    cdef double iS22
    cdef double iS33
    cdef double iS01
    cdef double iS02
    cdef double iS03
    cdef double iS12
    cdef double iS13
    cdef double iS23
    cdef double d0
    cdef double d1
    cdef double d2
    cdef double d3
    # cdef double u_m_0
    # cdef double u_m_1
    # cdef double u_m_2
    # cdef double u_m_3


    #with nogil:
    for 0 <= m < M-1:
        mkk  = m*k*k
        mN   = m*N
        #mNk   = m*N*k
        #mNk   = m*N*k
        mK =   m*k

        #  write output to p_qdr[m, n].  (xn - um)iSg_m (xn - um)

        iS00 = p_iSg[mkk]
        iS11 = p_iSg[mkk+k+1]
        iS22 = p_iSg[mkk+2*k+2]
        iS33 = p_iSg[mkk+3*k+3]
        iS01 = 2*p_iSg[mkk+1]   #  this
        iS02 = 2*p_iSg[mkk+2]
        iS03 = 2*p_iSg[mkk+3]
        iS12 = 2*p_iSg[mkk+k+2]
        iS13 = 2*p_iSg[mkk+k+3]
        iS23 = 2*p_iSg[mkk+2*k+3]

        u_m_0 = p_u[mK]
        u_m_1 = p_u[mK + 1]
        u_m_2 = p_u[mK + 2]
        u_m_3 = p_u[mK + 3]


        pfxm = p_fx[m]
        piqx2m= p_iqx2[m]
        pfym = p_fy[m]
        piqy2m= p_iqy2[m]

        pkFR_m = p_pkFR[m]
        mkNrms_m = p_mkNrms[m]

        for n in range(N):
            #mNk_nk = mNk + n*k
            mNn    = mN+n
            nK     = n*k

            d0 = p_mAS[nK]-p_u[mK]
            d1 = p_mAS[nK+1]-p_u[mK+1]
            d2 = p_mAS[nK+2]-p_u[mK+2]
            d3 = p_mAS[nK+3]-p_u[mK+3]               
            #                p_qdrMKS[mNn] = d0*(iS00*d0 + iS01*d1 + iS02*d2 + iS03*d3)+\
            s_qdrMKS = d0*(iS00*d0 + iS01*d1 + iS02*d2 + iS03*d3)+\
                d1*(iS11*d1 + iS12*d2 + iS13*d3) +\
                d2*(iS22*d2 + iS23*d3) +\
                d3*iS33*d3

            #p_qdrSPC[mNn] = (pfm - p_xAS[n])*(pfm - p_xAS[n])*piq2m
            s_qdrSPC = (pfxm - p_xAS[n])*(pfxm - p_xAS[n])*piqx2m + (pfym - p_yAS[n])*(pfym - p_yAS[n])*piqy2m

            #p_exp_arg[mNn] = pkFR_m + mkNrms_m - 0.5*(p_qdrSPC[mNn] + p_qdrMKS[mNn])
            s_exp_arg      = pkFR_m + mkNrms_m - 0.5*(s_qdrSPC + s_qdrMKS)
            if s_exp_arg > p_max_so_far[n]:
                p_max_so_far[n] = s_exp_arg
            p_exp_arg[mNn] = s_exp_arg


    m    = M-1
    mkk  = m*k*k
    mN   = m*N
    #mNk   = m*N*k
    #mNk   = m*N*k
    mK =   m*k

    #  write output to p_qdr[m, n].  (xn - um)iSg_m (xn - um)

    iS00 = p_iSg[mkk]
    iS11 = p_iSg[mkk+k+1]
    iS22 = p_iSg[mkk+2*k+2]
    iS33 = p_iSg[mkk+3*k+3]
    iS01 = 2*p_iSg[mkk+1]   #  this
    iS02 = 2*p_iSg[mkk+2]
    iS03 = 2*p_iSg[mkk+3]
    iS12 = 2*p_iSg[mkk+k+2]
    iS13 = 2*p_iSg[mkk+k+3]
    iS23 = 2*p_iSg[mkk+2*k+3]

    u_m_0 = p_u[mK]
    u_m_1 = p_u[mK + 1]
    u_m_2 = p_u[mK + 2]
    u_m_3 = p_u[mK + 3]


    pfxm = p_fx[m]
    piqx2m= p_iqx2[m]
    pfym = p_fy[m]
    piqy2m= p_iqy2[m]

    pkFR_m = p_pkFR[m]
    mkNrms_m = p_mkNrms[m]

    for n in range(N):
        #mNk_nk = mNk + n*k
        mNn    = mN+n
        nK     = n*k

        d0 = p_mAS[nK]-p_u[mK]
        d1 = p_mAS[nK+1]-p_u[mK+1]
        d2 = p_mAS[nK+2]-p_u[mK+2]
        d3 = p_mAS[nK+3]-p_u[mK+3]               
        #p_qdrMKS[mNn] = d0*(iS00*d0 + iS01*d1 + iS02*d2 + iS03*d3)+\
        s_qdrMKS = d0*(iS00*d0 + iS01*d1 + iS02*d2 + iS03*d3)+\
            d1*(iS11*d1 + iS12*d2 + iS13*d3) +\
            d2*(iS22*d2 + iS23*d3) +\
            d3*iS33*d3

        #p_qdrSPC[mNn] = (pfm - p_xAS[n])*(pfm - p_xAS[n])*piq2m
        s_qdrSPC = (pfxm - p_xAS[n])*(pfxm - p_xAS[n])*piqx2m + (pfym - p_yAS[n])*(pfym - p_yAS[n])*piqy2m        
        #s_qdrSPC = (pfm - p_xAS[n])*(pfm - p_xAS[n])*piq2m

        #p_exp_arg[mNn] = pkFR_m + mkNrms_m - 0.5*(p_qdrSPC[mNn] + p_qdrMKS[mNn])
        s_exp_arg = pkFR_m + mkNrms_m - 0.5*(s_qdrSPC + s_qdrMKS)

        if s_exp_arg > p_max_so_far[n]:
            p_max_so_far[n] = s_exp_arg
        p_exp_arg[mNn] = s_exp_arg                

        for 0 <= mm < M:
            p_econt[mm*N+n] = exp(p_exp_arg[mm*N+n]-p_max_so_far[n])



            
    for 0 <= m < M:
        mN = m*N
        for 0 <= n < N:
            mNpn = mN+n
            #  [m+1, n]  = [m, n] + [m, n]
            #  (m+1)*N + n = mN + n + N
            #p_rat[mNpn + N] = p_rat[mNpn] + p_econt[mNpn]
            #p_rat[(m+1)*N + n] = p_rat[mNpn] + p_econt[mNpn]
            rat[m+1, n] = rat[m, n] + econt[m, n]

    for 0 <= n < N:   #  rat[M, n] = M*N + n
        #  rnds[0] *= rat[M, 0]
        #     p_rnds[n] *= p_rat[M*N + 0]
        #  rnds[1] *= rat[M, 1]
        #     p_rnds[n] *= p_rat[M*N + 1]            
        #p_rnds[n] *= p_rat[M*N+n]
        rnds[n] *= rat[M, n]
                

    # printf("%.4e\n", p_rat[M*N])
    # print("%.4e" % rat[M, 0])
    # printf("%.4e\n", p_rat[M*N+5])
    # print("%.4e" % rat[M, 5])    
    # printf("%.4e\n", p_rat[M*N+N-1])
    # print("%.4e" % rat[M, N-1])    
    

@cython.boundscheck(False)
@cython.wraparound(False)
def set_occ_cgz(long[::1] clstsz, double[:, ::1] crats, double[::1] rnds, unsigned char[::1] cgz, long M, long N):
    #  instead of doing the following:
    #M1 = crat[1:] >= rnds
    #M2 = crat[0:-1] <= rnds
    #gz = (M1&M2)
    # in python to occupation binary vector gz with 0 or 1s,
    #  call this function (with call to gz.fill(0) before calling set_occ)
    cdef long n, im, #nind, m
    cdef double* p_crats = &crats[0, 0]
    cdef long*   p_clstsz= &clstsz[0]
    cdef double* p_rnds  = &rnds[0]
    cdef unsigned char* p_cgz = &cgz[0]   #  N x M    different than c_rats
    cdef double rnd

    with nogil:
        for m in range(M):
            p_clstsz[m] = 0
        for n in range(N):
            im = 0
            while p_rnds[n] >= p_crats[im*N+n]:   #  crats
            #while rnds[n] >= crats[im, n]:   #  crats
                # rnds[0] = 0.1  crats = [0, 0.2]    i expect gz[0, n] = 1
                im += 1
            p_cgz[n] = im-1
            #cgz[n] = im-1
            #p_cgz[n*M] = im
            p_clstsz[im-1] += 1
            #p_clstsz[im] += 1

                #  actually slower than calling gz.fill(0) before call to srch_occ
                # for im*N+n <= m < M*N+n by N:
                #     p_gz[m] = 0

                
def multiple_mat_dot_v(double[:, :, ::1] mat, double[:, ::1] vec, double[:, ::1] out, long M, long K):
    ##  mat is M x K x K
    ##  vec is M x K
    ##  out is M x K
    cdef long m, k, mKK, mK, iK, i, j
    cdef double* p_mat = &mat[0, 0, 0]
    cdef double* p_vec = &vec[0, 0]
    cdef double* p_out = &out[0, 0]

    with nogil:
        for 0 <= m < M:
            mKK = m*K*K
            mK  = m*K
            for i in range(K):
                iK = i*K
                p_out[mK+ i] = 0
                for k in range(K):
                    p_out[mK+ i] += p_mat[mKK+ iK+ k] * p_vec[mK + k]


def Sg_PSI(long[::1] cls_str_ind, long[::1] clstsz, long[::1] v_sts, double[:, ::1] mks, double[:, :, ::1] _Sg_PSI, double[:, :, ::1] Sg_PSI_, double[:, ::1] u, long M, long K):
    cdef long non_cnt_ind, m, k, n, nSpks, mK, i0, mK2
    cdef long* p_cls_str_ind = &cls_str_ind[0]
    cdef long* p_clstsz      = &clstsz[0]
    cdef long* p_v_sts       = &v_sts[0]
    cdef double* p_mks       = &mks[0, 0]
    cdef double* p_u         = &u[0, 0]
    cdef double* p_Sg_PSI_   = &Sg_PSI_[0, 0, 0]
    cdef double* p__Sg_PSI   = &_Sg_PSI[0, 0, 0]
    cdef long K2 = K*K
    cdef double tot11, tot12, tot13, tot14, tot22, tot23, tot24, tot33, tot34, tot44
    cdef double dv1, dv2, dv3, dv4

    with nogil:
        for 0 <= m < M:
            nSpks = p_cls_str_ind[m+1] - p_cls_str_ind[m]
            i0    = p_cls_str_ind[m]
            mK    = m*K
            mK2   = m*K2

            tot11 = 0
            tot12 = 0
            tot13 = 0
            tot14 = 0
            tot22 = 0
            tot23 = 0
            tot24 = 0
            tot33 = 0
            tot34 = 0
            tot44 = 0            

            for 0 <= n < nSpks:
                non_cnt_ind = p_v_sts[i0 + n]
                dv1 = p_mks[non_cnt_ind*K]-p_u[mK]
                dv2 = p_mks[non_cnt_ind*K+1]-p_u[mK+1]
                dv3 = p_mks[non_cnt_ind*K+2]-p_u[mK+2]
                dv4 = p_mks[non_cnt_ind*K+3]-p_u[mK+3]                
                tot11 += dv1*dv1
                tot12 += dv1*dv2
                tot13 += dv1*dv3
                tot14 += dv1*dv4
                ##
                tot22 += dv2*dv2
                tot23 += dv2*dv3
                tot24 += dv2*dv4
                ##
                tot33 += dv3*dv3
                tot34 += dv3*dv4
                ##
                tot44 += dv4*dv4
            #  Sg_PSI_[m, 0, 0] m*K2
            #  Sg_PSI_[m, 0, 1] m*K2+1
            #  Sg_PSI_[m, 0, 2] m*K2+2
            #  Sg_PSI_[m, 0, 3] m*K2+3
            #  Sg_PSI_[m, 1, 0] m*K2+K
            #  Sg_PSI_[m, 1, 1] m*K2+K+1
            #  Sg_PSI_[m, 1, 2] m*K2+K+2
            #  Sg_PSI_[m, 1, 3] m*K2+K+3
            #  Sg_PSI_[m, 2, 0] m*K2+2*K
            #  Sg_PSI_[m, 2, 1] m*K2+2*K+1
            #  Sg_PSI_[m, 2, 2] m*K2+2*K+2
            #  Sg_PSI_[m, 2, 3] m*K2+2*K+3
            #  Sg_PSI_[m, 3, 0] m*K2+3*K
            #  Sg_PSI_[m, 3, 1] m*K2+3*K+1
            #  Sg_PSI_[m, 3, 2] m*K2+3*K+2
            #  Sg_PSI_[m, 3, 3] m*K2+3*K+3
            
            p_Sg_PSI_[mK2] = tot11   + p__Sg_PSI[mK2]
            p_Sg_PSI_[mK2+1] = tot12 + p__Sg_PSI[mK2+1]
            p_Sg_PSI_[mK2+2] = tot13 + p__Sg_PSI[mK2+2]
            p_Sg_PSI_[mK2+3] = tot14 + p__Sg_PSI[mK2+3]
            ####
            p_Sg_PSI_[mK2+4] = tot12 + p__Sg_PSI[mK2+4] # tot21
            p_Sg_PSI_[mK2+5] = tot22 + p__Sg_PSI[mK2+5]
            p_Sg_PSI_[mK2+6] = tot23 + p__Sg_PSI[mK2+6]
            p_Sg_PSI_[mK2+7] = tot24 + p__Sg_PSI[mK2+7]
            ####
            p_Sg_PSI_[mK2+8] = tot13 + p__Sg_PSI[mK2+8] #  = tot31
            p_Sg_PSI_[mK2+9] = tot23 + p__Sg_PSI[mK2+9] #  = tot32
            p_Sg_PSI_[mK2+10]= tot33 + p__Sg_PSI[mK2+10]
            p_Sg_PSI_[mK2+11]= tot34 + p__Sg_PSI[mK2+11]
            ####
            p_Sg_PSI_[mK2+12]= tot14 + p__Sg_PSI[mK2+12] #  = tot41
            p_Sg_PSI_[mK2+13]= tot24 + p__Sg_PSI[mK2+13] #  = tot42
            p_Sg_PSI_[mK2+14]= tot34 + p__Sg_PSI[mK2+14] #  = tot43
            p_Sg_PSI_[mK2+15]= tot44 + p__Sg_PSI[mK2+15]           
            
            # for 0 <= k < K:
            #     uk = p_u[mK+k]
            #     tot = 0


                # for 0 <= n < nSpks:
                #     non_cnt_ind = p_v_sts[i0 + n]

                #     tot += (p_mks[non_cnt_ind*K + k]-uk)*(p_mks[non_cnt_ind*K + k]-uk)
                # p_Sg_PSI_[m*K2+k*K + k] = p__Sg_PSI[m*K2+k*K + k] + tot

@cython.cdivision(True)
def find_mcs(long[::1] clstsz, long[::1] v_sts, long[::1] cls_str_ind, double[:, ::1] mks, double [:, ::1] mcs, long M_use, long K):
    cdef long m, n, nSpks, i0, mK, k
    cdef long* p_clstsz = &clstsz[0]
    cdef double* p_mcs   = &mcs[0, 0]
    cdef double* p_mks   = &mks[0, 0]
    cdef long* p_v_sts   = &v_sts[0]
    cdef long* p_cls_str_ind   = &cls_str_ind[0]

    with nogil:
        for 0 <= m < M_use:
            nSpks = p_cls_str_ind[m+1] - p_cls_str_ind[m]
            i0    = p_cls_str_ind[m]
            mK    = m*K

            for 0 <= k < K:
                p_mcs[mK+k] = 0
            for 0 <= n < nSpks:
                #  elapsed time ratios
                for 0 <= k < K:
                    p_mcs[mK+k] += p_mks[p_v_sts[i0+n]*K + k]

            if nSpks > 0:
                for 0 <= k < K:
                    p_mcs[mK+k] /= nSpks



def cluster_bounds2_cgz(long[::1] clstsz, long[::1] Asts, long[::1] cls_str_ind, long[::1] v_sts, unsigned char[::1] cgz, long t0, long M_use, long N):
    ###############  FOR EACH CLUSTER
    cdef long i0 = 0
    cdef long[::1] mv_minds
    cdef long* p_minds
    cdef long* p_clstsz = &clstsz[0]
    cdef long* p_cls_str_ind = &cls_str_ind[0]
    cdef long* p_v_sts = &v_sts[0]
    cdef long* p_Asts = &Asts[0]
    cdef unsigned char* p_cgz   = &cgz[0]
    cdef long ns, n, m
    
    p_cls_str_ind[0]         = i0

    with nogil:
        for 0 <= m < M_use:   #  get the minds
            p_cls_str_ind[m+1]         = i0 + p_clstsz[m]
            if p_clstsz[m] > 0:
                ns = 0
                for 0 <= n < N:
                    if p_cgz[n] == m:
                        p_v_sts[i0+ns] = p_Asts[n] + t0
                        ns += 1
            i0 += p_clstsz[m]
            
