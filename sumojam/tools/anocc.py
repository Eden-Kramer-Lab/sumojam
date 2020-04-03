#import sumojam.tools.linearize_funcs as _lfuncs
import numpy as _N
import matplotlib.pyplot as _plt
#from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
import time as _tm

def gauKer(w):
    wf = _N.empty(8*w+1)

    for i in range(-4*w, 4*w+1):
        wf[i+4*w] = _N.exp(-0.5*(i*i)/(w*w))

    return wf


def approx_maze_sungod(dat):
    change_seg = _N.where(_N.diff(dat[:, 7]) != 0)[0]

    Nsgs          = 8

    seg_chgs        = _N.where(_N.diff(dat[:, 7]) != 0)[0]
    seg_strtend     = _N.empty((len(seg_chgs)+1, 2), dtype=_N.int)

    for ip in range(len(seg_chgs)+1):
        seg_strtend[ip, 0] = 0 if (ip == 0) else seg_chgs[ip-1]+1
        seg_strtend[ip, 1] = seg_chgs[ip] if (ip < len(seg_chgs)) else dat.shape[0]

    random_state = 3
    tot_t = 0
    #for ip in range(-1, 8):

    mns_all = []
    sd2s_all = []
    wgts_all = []
    Ns_all = []

    print(seg_strtend)
    iip = 0
    for ip in range(-1, 8):
        n_components = 30 if ip == -1 else 15

        mving_in_this_seg = _N.where((dat[:, 7] == ip) & (dat[:, 8] == 1))[0]

        if len(mving_in_this_seg) > 0:
            iip += 1

            #  3000 pts at most.  
            skp = int(len(mving_in_this_seg) / 3000)
            #  2200   skp is 1.
            #  3200   skp is 1.  (upto 6000 points)
            #  6000   skp is 2.  (number of points used starts back at 3000)
            
            sk  = 1 if skp == 0 else skp

            X = _N.array(dat[mving_in_this_seg[::skp], 0:2])

            # bgm = BayesianGaussianMixture(
            #     weight_concentration_prior_type="dirichlet_process",
            #     weight_concentration_prior=0.8,
            #     degrees_of_freedom_prior=12,
            #     n_components=n_components, reg_covar=0, init_params='random',
            #     max_iter=1500, mean_precision_prior=.8,
            #     random_state=random_state, covariance_type="diag")

            ICs     = 5
            em_mns  = _N.empty((ICs, n_components, 2))
            em_sd2s = _N.empty((ICs, n_components, 2))
            em_wgts = _N.empty((ICs, n_components))
            lklhd_ubs= _N.empty(ICs)

            ###  run EM several times
            for ic in range(ICs):
                random_state = int(1000*_N.random.rand())
                
                emgm = GaussianMixture\
                    (n_components=n_components, init_params='kmeans',\
                     #means_init=mn_init,\
                     max_iter=1500,\
                     random_state=random_state, covariance_type="diag")
                emgm.fit(X)
                em_mns[ic]  = emgm.means_
                em_sd2s[ic] = emgm.covariances_
                em_wgts[ic] = emgm.weights_
                lklhd_ubs[ic] = emgm.lower_bound_

            t4  = _tm.time()

            ###
            #  calculate estimated pdf using parameters found via EM
            ###
            bestIC = _N.where(lklhd_ubs == _N.max(lklhd_ubs))[0][0]
            mns_r  = em_mns[bestIC].T.reshape((2, n_components))
            isd2s_r= (1./em_sd2s[bestIC]).T.reshape((2, n_components))
            sd2s_r = em_sd2s[bestIC].T.reshape((2, n_components))

            srtdinds = _N.argsort(emgm.weights_)
            #  99%
            totw = 0
            iw    = len(srtdinds)
            ncusd= 0
            #print(bgm.weights_)
            while totw < 0.999:
                iw -= 1
                totw += emgm.weights_[iw]
                ncusd += 1

            #print("cmps %(c)d   tm %(t).2f" % {"c" : ncusd, "t" : (t2-t1)})


            mns_all.extend(_N.array(em_mns[bestIC][srtdinds[n_components-ncusd:]]))
            sd2s_all.extend(_N.array(em_sd2s[bestIC][srtdinds[n_components-ncusd:]]))
            wgts_all.extend(len(mving_in_this_seg)*_N.array(em_wgts[bestIC][srtdinds[n_components-ncusd:]]))
            #  wgts_all   -->  when summed should equal total time in # of dt's

    return len(mns_all), _N.array(mns_all), _N.array(sd2s_all), _N.array(wgts_all)


def split_up_1d(dat, mving, xbns, smth_krnl):
    _occ_cnts, bnsx = _N.histogram(dat[mving, 0], bins=xbns)
    if smth_krnl > 0:    #  SMOOTHING DONE TO FIND PEAKS
        gk     = gauKer(smth_krnl)
        gk     /= _N.sum(gk)

        occ_cnts = _N.convolve(_occ_cnts, gk, mode="same")
    else:
        occ_cnts = _occ_cnts

    ###  count number of peaks in histogram for approximate # of GMM components
    docc   = _N.diff(occ_cnts)
    peaks  = _N.where((docc[0:-1] > 0) & (docc[1:] < 0))[0]
    print("peaks %d" % len(peaks))
    print(peaks)
    n_components = int(len(peaks) * 3)
    print("n_components %d" % n_components)

    #  3 peaks at a time


    nCmps = []
    brdrs = []
    pks   = []      #  for initing
    iLast = 0
    byPks = 3

    print("****")
    for i in range(0, len(peaks), byPks):  #  put one between i-1 and i
        if (i > 0):# and (peaks[i] - peaks[i-1] > ):
            brdrs.append(int(0.5*(peaks[i-1] + peaks[i])))
            nCmps.append(i - iLast)
            #print(peaks[iLast:i])
            #pks.append(peaks[iLast:i])
            iLast = i

    if (len(peaks) % byPks) == 0:
        nCmps.append(byPks)
        #print(peaks[iLast:len(peaks)])                            
    else:
        nCmps.append(len(peaks) - iLast)
        #print(peaks[iLast:len(peaks)])                    
    print("****")            
    #return brdrs, pks, nCmps
    return brdrs, nCmps
    

def approx_maze_1D_smthd(dat, BNS=30, smth_krnl=2, nz_pth=None):
    """
    bin the path data to first find the peaks.  We want a relatively high number
    of bins (BNS), but we smoothed the binned data using (smth_krnl) to keep 
    the number of candidate bins to be fairly low.

    We should also add noise to the path, to give a bit of width to actual
    areas of high occupancy.  
    """
    random_state = 3
    tot_t = 0

    mns_all = []
    sd2s_all = []
    wgts_all = []
    Ns_all = []

    iip = 0

    #  x  01  m1 m2 m3 m4 01
    
    mving = _N.arange(dat.shape[0])
    max_x = _N.max(dat[mving, 0])
    min_x = _N.min(dat[mving, 0])
    amp_x = max_x - min_x


    ###  build histogram of data, need this to 
    xbns   = _N.linspace(min_x-0.001*amp_x, max_x + 0.001*amp_x, BNS+1)
    xms    = 0.5*(xbns[0:-1] + xbns[1:])
    xms_r  = xms.reshape((BNS, 1))
    
    dx     = _N.diff(xbns)[0]
    
    _occ_cnts, bnsx = _N.histogram(dat[mving, 0], bins=xbns)
    if smth_krnl > 0:
        gk     = gauKer(smth_krnl)
        gk     /= _N.sum(gk)

        occ_cnts = _N.convolve(_occ_cnts, gk, mode="same")
    else:
        occ_cnts = _occ_cnts
    #fig = _plt.figure()
    print("smth_krnl  %d" % smth_krnl)
    #_plt.plot(xms, (_occ_cnts/mving.shape[0]), color="green")
    #_plt.plot(xms, (occ_cnts/mving.shape[0]), color="orange")

    #brdrs, peaks, nComponents = split_up_1d(dat, mving, xbns, smth_krnl)
    brdrs, nComponents = split_up_1d(dat, mving, xbns, smth_krnl)

    brdrsWE = [0] + brdrs + [BNS]
    a_brdrsWE = _N.array(brdrsWE)
    print(_N.diff(a_brdrsWE))

    N      = dat.shape[0]

    print(brdrsWE)

    #fig = _plt.figure(figsize=(4, 3))    
    for pc in range(len(brdrsWE)-1):
        indcs = _N.where((dat[mving, 0] > xbns[brdrsWE[pc]]) & (dat[mving, 0] < xbns[brdrsWE[pc+1]]))[0]

        print("low %(l).3f    hi %(h).3f" % {"l" : xbns[brdrsWE[pc]], "h" : xbns[brdrsWE[pc+1]]})

        #_plt.plot(dat[mving[indcs], 0])
        
        cN    = int(30000*(float(len(indcs)) / N))
        n_components = nComponents[pc]*3
        if len(mving) > 0:
            these = _N.random.choice(indcs, cN)
            print("....  %d"  % cN)
            #  3000 pts at most.  

            dat[mving[these], 0] += nz_pth*_N.random.randn(cN)            
            X = _N.array(dat[mving[these], 0]).reshape((cN, 1))

            #fig = _plt.figure()
            #_plt.hist(X, bins=_N.linspace(-6, 6, 241))
            ICs     = 2
            em_mns  = _N.empty((ICs, n_components, 1))
            em_sd2s = _N.empty((ICs, n_components, 1))
            em_wgts = _N.empty((ICs, n_components))
            lklhd_ubs= _N.empty(ICs)

            print("n_components %d" % n_components)
            ###  run EM several times
            # print(_N.array(peaks[pc], dtype=_N.int))
            # mn_init = _N.zeros((n_components, 1))
            # mn_init[0:len(peaks[pc]), 0] = xbns[_N.array(peaks[pc], dtype=_N.int)]
            # print(mn_init)

            for ic in range(ICs):
                random_state = int(10000*_N.random.rand())

                emgm = GaussianMixture\
                    (n_components=n_components, init_params='kmeans',\
                     max_iter=5000,\
                     #means_init=mn_init,\
                     tol=1e-5,\
                     random_state=random_state, covariance_type="diag")
                emgm.fit(X)
                em_mns[ic]  = emgm.means_
                em_sd2s[ic] = emgm.covariances_
                em_wgts[ic] = emgm.weights_
                lklhd_ubs[ic] = emgm.lower_bound_

            #emgm.fit(X)
            srtdinds = _N.argsort(emgm.weights_)

            bestIC = _N.where(lklhd_ubs == _N.max(lklhd_ubs))[0][0]
            mns_r  = em_mns[bestIC].T.reshape((1, n_components))
            isd2s_r= (1./em_sd2s[bestIC]).T.reshape((1, n_components))
            sd2s_r = em_sd2s[bestIC].T.reshape((1, n_components))
            mns  = em_mns[bestIC]
            isd2s= (1./em_sd2s[bestIC])
            sd2s = em_sd2s[bestIC]

            #  99%
            totw = 0
            iw    = len(srtdinds)
            ncusd= 0

            while totw < 0.99:     #  sort by weights
                iw -= 1
                totw += emgm.weights_[iw]
                ncusd += 1
            print("!!!!!!!!!!   ncusd %d" % ncusd)

            t2 = _tm.time()

            A      = (em_wgts[bestIC] / _N.sqrt(2*_N.pi*sd2s_r)) * dx
            occ_x = _N.sum(A*_N.exp(-0.5*(xms_r - mns_r)*(xms_r - mns_r)*isd2s_r), axis=1)

            print(emgm.covariances_.shape)
            #mns_all.extend(_N.array(mns_r[0, srtdinds[n_components-ncusd:]]).tolist())
            #sd2s_all.extend(_N.array(emgm.covariances_[srtdinds[n_components-ncusd:], 0]).tolist())
            #wgts_all.extend((len(indcs)*_N.array(emgm.weights_[srtdinds[n_components-ncusd:]])).tolist())
            mns_all.extend(em_mns[bestIC])
            sd2s_all.extend(em_sd2s[bestIC])
            wgts_all.extend((len(indcs)*em_wgts[bestIC]))

            #  wgts_all   -->  when summed should equal total time in # of dt's

            # fig = _plt.figure(figsize=(4, 6))
            # fig.add_subplot(2, 1, 1)
            # _plt.plot(xms, occ_x)
            # fig.add_subplot(2, 1, 2)
            # _plt.hist(X, bins=_N.linspace(-6, 6, 241))

    #return len(mns_all[0]), _N.array(mns_all), _N.array(sd2s_all), _N.array(wgts_all)

    return len(mns_all), _N.array(mns_all)[:, 0], _N.array(sd2s_all)[:, 0], _N.array(wgts_all)


# def GKs(dat, binw):
#     print("anocc.py GKs() called")
#     Nsgs          = 8

#     tot_t = 0
#     #for ip in range(-1, 8):

#     mns_all = []
#     sd2s_all = []
#     wgts_all = []
#     Ns_all = []

#     for ip in range(-1, 8):
#         n_bins = 20 if ip == -1 else 25

#         #mving_in_this_seg = _N.where((dat[:, 7] == ip) & (dat[:, 8] == 1))[0]
#         mving_in_this_seg = _N.where((dat[:, 7] == ip))[0]  #  should just be "this seg"

#         if len(mving_in_this_seg) > 0:
#             x_mvg = _N.array(dat[mving_in_this_seg, 0])
#             y_mvg = _N.array(dat[mving_in_this_seg, 1])
#             xMax  = _N.max(x_mvg)
#             xMin  = _N.min(x_mvg)
#             xA    = xMax - xMin
#             yMax  = _N.max(y_mvg)
#             yMin  = _N.min(y_mvg)
#             yA    = yMax - yMin
#             n_bins_x = int(_N.ceil((1.02*xA) / binw) + 1)
#             n_bins_y = int(_N.ceil((1.02*yA) / binw) + 1)
#             print("n_bins_x   %(x)d     n_bins_y   %(y)d" % {"x" : n_bins_x, "y" : n_bins_y})
#             xms   = _N.linspace(xMin-0.01*xA, xMax+0.01*xA, n_bins_x+1)
#             yms   = _N.linspace(yMin-0.01*yA, yMax+0.01*yA, n_bins_y+1)

#             #  area underneath  is 1
#             fig = _plt.figure()
#             cnts, bins_x, bins_y = _N.histogram2d(x_mvg, y_mvg, bins=[xms, yms])

#             nz_bxs, nz_bys = _N.where(cnts > 0)
#             mns = _N.empty((len(nz_bxs), 2))
#             sd2s = _N.empty((len(nz_bxs), 2))
#             wgts = _N.empty(len(nz_bxs))

#             hf_dx = _N.diff(xms)[0]*0.5
#             hf_dy = _N.diff(yms)[0]*0.5
#             #hf_dx = _N.diff(xms)[0]
#             #hf_dy = _N.diff(yms)[0]
            
            
#             for ii in range(len(nz_bxs)):
#                 ix = nz_bxs[ii]
#                 iy = nz_bys[ii]

#                 #  area underneath is A/sqrt(4*_N.pi*hf_dx*hf_dx*hf_dy*hf_dy) = cn
#                 cn = cnts[ix, iy]
#                 A  = cn#/(2*_N.pi*_N.sqrt(hf_dx*hf_dx*hf_dy*hf_dy))
#                 wgts[ii] = A
#                 mns[ii, 0] = xms[ix]
#                 mns[ii, 1] = yms[iy]
#                 sd2s[ii, 0] = hf_dx*hf_dx
#                 sd2s[ii, 1] = hf_dy*hf_dy
#             mns_all.extend(mns)
#             sd2s_all.extend(sd2s)
#             wgts_all.extend(wgts)            

#     #return len(mns_all), _N.array(mns_all), _N.array(sd2s_all), _N.array(wgts_all)
#     #return mns_all, sd2s_all, wgts_all
