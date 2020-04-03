import numpy as _N
cimport numpy as _N
from libc.math cimport sqrt
cimport cython



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def invwishartrand_multi_tempM(int M, int K, nu, double[:, :, ::1] PSI):
    f = _N.zeros((M, K, K))
    b = _N.zeros((M, K, K))
    #ib = _N.zeros((M, K, K))
    r = _N.zeros((M, K, K))     #  return value

    nu_da = nu#+K - 1    #  nu dimension-adjusted (don't need to do this - doing this matches generated output to that of _ss.invwishart.rvs)
    cdef double[:, :, ::1] mv_f = f
    cdef double* p_f         = &mv_f[0, 0, 0]
    cdef double[:, :, ::1] mv_b = b
    cdef double* p_b         = &mv_b[0, 0, 0]
    cdef double[:, :, ::1] mv_r  = r   
    cdef double* p_r_all         = &mv_r[0, 0, 0]
    cdef double* p_r
    cdef int _2K = 2*K
    cdef int _3K = 3*K

    Sinvchol         = _N.linalg.cholesky(_N.linalg.inv(PSI))
    cdef double[:, :, ::1] mv_Sinvchol = Sinvchol
    cdef double* p_Sinvchol_all         = &mv_Sinvchol[0, 0, 0]
    cdef double* p_Sinvchol

    #  number of Gauss rands needed for lower-triangular:  K(K-1) / 2
    #  number of chisquare rands needed:  K
    #  fill the lower-triangular matrix

    #  (K*(K-1)/2) == n_tril  (_ss.stats._multivariate)
    g_rands  = _N.random.randn(M*(K*(K-1)/2))
    cdef double[::1] mv_g_rands         = g_rands
    cdef double* p_g_rands         = &mv_g_rands[0]

    #  arranged (M, K).  read the flattened array 
    nu_da_r  = nu_da.reshape((M, 1))
    _0_K     = _N.arange(K)
    _0_K_r   = _0_K.reshape((1, K))
    #X2_rands = _N.sqrt(_N.random.chisquare(nu_da_r-_0_K_r))
    sqrt_X2_rands = _N.sqrt(_N.random.chisquare(nu_da_r-_0_K_r))

    #print((nu_da_r-_0_K_r))
    cdef double[:, ::1] mv_sqrt_X2_rands         = sqrt_X2_rands
    cdef double* p_sqrt_X2_rands         = &mv_sqrt_X2_rands[0, 0]

    cdef int ir_g = 0
    cdef int ir_x2= 0
    cdef double total
    cdef int i, j, k, l, m

    cdef int K2m
    #  f is 'A' in (_ss.stats._multivariate)
    with nogil:
        for m in range(M):
            K2m = K*K*m
            p_Sinvchol   = &(p_Sinvchol_all[K2m])            
            for i in range(K):
                for j in range(i):
                    p_f[K2m+K*i+j] = p_g_rands[ir_g]
                    ir_g += 1
                p_f[K2m+K*i+i] = p_sqrt_X2_rands[ir_x2]
                ir_x2 += 1

#
            #for i in range(K):
                for j in range(i+1):
                    total = 0
                    for k in range(j, i+1):
                        total += p_Sinvchol[i*K + k] * p_f[K2m+k*K + j]

                    p_b[K2m+K*i+ j] = total

    #return f
    ib   = _N.linalg.inv(b)

    cdef double[:, :, ::1] mv_ib = ib
    cdef double* p_ib         = &mv_ib[0, 0, 0]

    #  b is a lower triangular matrix
    #  _N.dot(b.T, b) is symmetric
    with nogil:
        for m in range(M):
            K2m = K*K*m
            for i in range(K):
                for j in range(i+1):
                    #  p_ib_T[i, l]*p_ib[l, j] =
                    #  p_ib[l, i]*p_ib[l, j]
                    total = 0
                    for l in range(K):
                        total += p_ib[K2m+l*K+i]*p_ib[K2m+l*K+j]
                    p_r_all[K2m+i*K+j] = total
                    p_r_all[K2m+j*K+i] = total
            #r[m]    = _N.dot(ib.T, ib)

    return r
