"""
kugupu
KUGUPUUU!
"""

# Make Python 2 and 3 imports work the same
# Safe to remove with Python 3-only code
from __future__ import absolute_import

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

import sys
from loguru import logger
logger.stop()
logger.start(sys.stderr, format="{time} {level} {message}", level="INFO")

from .traj_io import KugupuResults, save_results, load_results
from . import time
from . import dimers
from . import yaehmop
from . import networks
from . import hamiltonian_reduce
from . import generate_results
from .generate_results import generate_H_frag_trajectory

# logger.start(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")
