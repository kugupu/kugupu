from libcpp.vector cimport vector

ctypedef vector[int] intvec
import cython

import numpy as np

@cython.boundscheck(False)
cdef void scan_pairing(
        double[:] couplings,
        double thresh,
        int n,
        intvec* durations,
        intvec* observations,
    ):
    """Scans a single coupling timeseries

    Parameters
    ----------
    couplings : float array
      timeseries of fragment coupling
    thresh : float
      critical value to consider coupling ON/alive
    n : int
      length of coupling
    durations, observations : vector[int]
      results arrays

    Returns
    -------
    None - modifies durations and observations in place
    """
    cdef int i, alive, duration

    # initially off
    alive = False
    for i in range(n):
        if not alive:
            if couplings[i] > thresh:
                # Turn on
                alive = True
                duration = 1
        else:
            if couplings[i] > thresh:
                # Stay on
                duration += 1
            else:
                # Turn off
                alive = False
                durations.push_back(duration)
                observations.push_back(1)
    if alive:
        # If on at end, mark as right censored
        durations.push_back(duration)
        observations.push_back(0)

@cython.boundscheck(False)
def determine_lifetimes(
        double[:, :, :] couplings,
        double thresh
    ):
    """Scan a coupling matrix timeseries to extract coupling durations

    Parameters
    ----------
    couplings : float64 array
      coupling values over time, with shape(nframes, nfrags, nfrags)
    thresh : float
      critical value above which a coupling is ON otherwise OFF

    Returns
    -------
    durations : np.ndarray dtype int
      the duration of each coupling in frames
    observations : np.ndarray dtype int
      for each duration, if the end of coupling was observed
      1=coupling finished, 0=coupling was right censored
    """
    cdef int nframes, nfrags, i
    cdef intvec durations, observations

    nframes = couplings.shape[0]
    nfrags = couplings.shape[1]
    durations = intvec()
    observations = intvec()
    # triu scan
    for i in range(nfrags):
        for j in range(i+1, nfrags):
            scan_pairing(
                couplings[:, i, j],
                thresh,
                nframes,
                & durations,
                & observations,
            )

    return np.asarray(durations), np.asarray(observations)
