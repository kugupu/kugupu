Visualising results
===================

visualising stuff in notebooks


NGLView troubleshooting
"""""""""""""""""""""""

The draw methods in ``kugupu.visualise`` use the nglview package within a Jupyter notebook,
occasionally this will not produce any output or viewbox.
To fix this,
from a command line terminal run "``nglview enable``".
This should produce some output ending with "Validating: OK".
Once this is done, make sure to restart the Jupyter notebook server,
(not just the notebook session) and the problem should be fixed.


API Reference
"""""""""""""

.. autofunction::
   kugupu.visualise.draw_network

.. autofunction::
   kugupu.visualise.draw_fragments
