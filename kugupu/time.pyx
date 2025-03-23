# cython: language_level=3, boundscheck=False, embedsignature=True
#    kugupu - molecular networks for change transport
#    Copyright (C) 2019  Micaela Matta and Richard J Gowers
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""kugupu.time

Determines time based characteristics of coupling networks

"""
import cython
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def determine_lifetimes(const double[:, :, :] coupling,
                        float thresh):
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
    cdef size_t i, j, k, n
    cdef int nframes, nfrags
    cdef object d, o
    cdef long[::1] d_v, o_v
    cdef long alive, duration

    nframes = coupling.shape[0]
    nfrags = coupling.shape[1]
    # maximum number of durations
    # nfrags(nfrags-1)//2 i-j combinations
    # nframes//2 possible lifes per pairing, ie T/F/T/F/T/F
    d = np.empty(nframes * nfrags * (nfrags-1) // 4, dtype=int)
    o = np.empty(nframes * nfrags * (nfrags-1) // 4, dtype=int)
    # memory views of Python objects
    d_v = d
    o_v = o

    n = 0
    # loop over triu of couplings
    for i in range(nfrags):
        for j in range(i+1, nfrags):
            alive = False
            # loop over timeseries of i-j coupling
            for k in range(nframes):
                if not alive:
                    if coupling[k, i, j] >= thresh:
                        # start of coupling
                        alive = True
                        duration = 1
                else:
                    if coupling[k, i, j] >= thresh:
                        # continue coupling
                        duration += 1
                    else:
                        # end of coupling
                        alive = False
                        d_v[n] = duration
                        o_v[n] = 1
                        n += 1
            # if alive at end, add duration but mark as unobserved
            if alive:
                d_v[n] = duration
                o_v[n] = 0
                n += 1
    # return only used portions of arrays
    return d[:n], o[:n]
