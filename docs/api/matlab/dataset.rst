Dataset orchestration
=====================

The workflow scripts create and retain a :mat:class:`STARMapDataset` value
object. See :doc:`../matlab` for dimensions, calling conventions, Snakemake
entry points and runtime requirements. The method help below lists the actual
parser options, including options that are accepted but unused.

State inventory
---------------

.. list-table:: Public properties
   :header-rows: 1
   :widths: 30 70

   * - Properties
     - Meaning
   * - ``inputPath``, ``outputPath``, ``fovID``, ``useGPU``
     - Input/output paths, FOV identifier and stored GPU flag.
   * - ``images``, ``projections``
     - Round-keyed dictionaries of 4-D arrays and projection cells.
   * - ``metadata``, ``layers``
     - Round dimensions/channel structs; sequencing, other, all and reference layer names.
   * - ``registration``
     - Reference images and global correction parameter structs; values differ by key/stage.
   * - ``signal``, ``codebook``
     - Spot tables/scores and gene-to-sequence/sequence-to-gene dictionaries.
   * - ``subtile``
     - Tile index (initially zero) and crop-coordinate table.
   * - ``jobToDo``, ``jobFinished``
     - Workflow bookkeeping; methods set completion fields. This is not a resume/checkpoint guarantee.

All 18 methods, including the constructor, are listed below. Typical processing
order is LoadRawImages, enhancement, registration, SpotFinding,
ReadsExtraction, LoadCodebook, ReadsFiltration, and SaveSignal. Set
``layers.ref`` after loading and before methods that default to that reference.
The class does not validate that every prerequisite stage has run.

.. mat:currentmodule:: .

.. mat:autoclass:: STARMapDataset
   :members:
   :member-order: bysource

.. mat:automethod:: STARMapDataset.STARMapDataset
