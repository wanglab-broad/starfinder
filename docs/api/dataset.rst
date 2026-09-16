starfinder.dataset
==================

Stateful orchestration and value types. Methods document state mutations and output paths. Image-processing methods require loaded four-dimensional rounds unless stated otherwise. See :doc:`contracts` and :doc:`backends`.

.. currentmodule:: starfinder.dataset

.. autosummary::
   :toctree: generated

   STARMapDataset
   FOV
   FOVPaths
   LayerState
   Codebook
   CropWindow
   SubtileConfig
   log_step

.. py:data:: Shift3D

   Alias of ``tuple[float, float, float]``: voxel components ``(dz, dy, dx)``; the producing function defines the sign.

.. py:data:: ImageArray

   Alias of ``numpy.ndarray`` for a ``(Z, Y, X, C)`` image; no runtime shape or dtype enforcement.

.. py:data:: ChannelOrder

   Alias of ``list[str]``: filename channel patterns in desired output channel order.
