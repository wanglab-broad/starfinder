Subtile and Fiji utilities
==========================

These project-owned helpers complete the top-level MATLAB inventory. They
support crop/stitch bookkeeping rather than molecular decoding. The inventory
covers all 28 top-level ``.m`` files (one class and 27 functions). The local
``GetExtents`` function inside ``ExtractFromLocation.m`` is an implementation
detail, not an independently callable interface. Third-party add-ons and
``workflow/scripts`` are excluded from generated API coverage.

.. mat:currentmodule:: .

.. mat:autofunction:: MakeSubtileTable

.. mat:autofunction:: ParseFijiTileConfiguration
