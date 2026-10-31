starfinder.io
=============

TIFF readers preserve dtype and require explicit selection for ambiguous axes.
See :doc:`contracts` for conversion, geometry and channel ordering.
The :doc:`output guide <../recipes>` shows loading and saving examples.

.. currentmodule:: starfinder.io

Loading
-------

.. autosummary::
   :toctree: generated

   ImageLoadConfig
   ImageLoadResult
   load_round
   load_volume
   load_volume_zyxc

Conversion and saving
---------------------

.. autosummary::
   :toctree: generated

   convert_image
   ImageConversionConfig
   save_volume

The chunked checkpoint of tiled execution
-----------------------------------------

The registered rounds of one FOV in one HDF5 file, written once by the global task of tiled execution and read by
window by its tile tasks (``CHUNK_EDGE`` is 128 voxels in Y and X).

.. autosummary::
   :toctree: generated

   CHUNK_EDGE
   chunked_round_sha256
   ChunkedCheckpointWriter
   read_chunked_record
   read_window
