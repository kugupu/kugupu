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
"""Save and load timeseries of Hamiltonians

"""
from collections import namedtuple
from datetime import datetime
import numpy as np
import h5py

from . import logger
from . import __version__


KugupuResults = namedtuple("KugupuResults",
                           ['frames',
                            'H_frag',
                            'degeneracy'])

_DATEFORMAT = '%Y-%m-%d %H:%M:%S'


def save_results(filename, results):
    """Save Kugupu results to HDF5 file

    Parameters
    ----------
    filename : str
      filename, must not yet exist.  '.hdf5' will be appended
      if not present
    results : kugupu.Results namedtuple
      finished results to save to file
    """
    if not filename.endswith('.hdf5'):
        filename += '.hdf5'

    logger.debug("Saving results to {}".format(filename))
    with h5py.File(filename, 'w-') as f:
        f.attrs['kugupu_version'] = __version__
        f.attrs['creation_date'] = datetime.now().strftime(_DATEFORMAT)

        f['frames'] = results.frames
        f['H_frag'] = results.H_frag
        f['degeneracy'] = results.degeneracy


def load_results(filename):
    """Load Kugupu results from HDF5 file

    Parameters
    ----------
    filename : str
      path to HDF5 file

    Returns
    -------
    results : KugupuResults namedtuple
      the contents of the file
    """
    if not filename.endswith('.hdf5'):
        filename += '.hdf5'

    logger.debug("Loading results from {}".format(filename))
    with h5py.File(filename, 'r+') as f:
        logger.debug("Results from {}"
                     "".format(datetime.strptime(f.attrs['creation_date'],
                                                 _DATEFORMAT)))
        logger.debug("Saved with version: {}"
                     "".format(f.attrs['kugupu_version']))
        idx = f['frames'][()]
        H_frag = f['H_frag'][()]
        deg = f['degeneracy'][()]

    return KugupuResults(
        frames=idx,
        H_frag=H_frag,
        degeneracy=deg,
    )


def concatenate_results(*results):
    """Concatenate two result sets

    Results must be provided in order, the frame indices won't
    be checked.

    Parameters
    ----------
    results : iterable of KugupuResults
      the results sets to combine, in order

    Returns
    -------
    results : KugupuResults
      combined results
    """
    frames = np.concatenate([r.frames for r in results])

    H_frag = np.concatenate([r.H_frag for r in results])

    return KugupuResults(
        frames=frames,
        H_frag=H_frag,
        degeneracy=results[0].degeneracy,
    )
