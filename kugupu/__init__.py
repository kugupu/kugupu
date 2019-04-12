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

def enable_debug_logging():
    """Increase amount of logging"""
    logger.stop()
    logger.start(sys.stderr, format="{time} {level} {message}", level="DEBUG")

def disable_debug_logging():
    """Use a normal amount of logging"""
    logger.stop()
    logger.start(sys.stderr, format="{time} {level} {message}", level="INFO")

disable_debugger()


from .results_io import KugupuResults, save_results, load_results
from . import time
from . import dimers
from . import _yaehmop
from . import _hamiltonian_reduce
from . import networks
from .generate_results import coupling_matrix
from . import cli
from . import visualise
# logger.start(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")
