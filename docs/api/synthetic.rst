starfinder.synthetic
====================

The :doc:`formed-amplicon specification <../synthetic-specification>` defines
``starfinder.synthetic/1`` and independent acceptance cases for the new
development model. ``generate_formed_scene`` implements its clean scene/truth
slice; the historical APIs retain their original semantics.

Pure processed-image scene generation. Importing this package does not import
benchmark orchestration. Generation returns arrays in memory; the benchmark
persistence adapter writes workflow TIFFs, JSON, scene tables and annotations.

.. currentmodule:: starfinder.synthetic

.. autosummary::
   :toctree: generated

   formed_scene_preset
   FormedScene
   FormedSceneConfig
   generate_codebook
   generate_dataset
   generate_displacement_field
   generate_formed_scene
   generate_registration_pairs
   generate_volume
   get_preset_config
   render_spots
   ScalarDistribution
   SyntheticConfig
   SyntheticDataset

Formed scenes
-------------

``formed_scene_preset(name)`` returns a supplied ``Codebook`` and
``FormedSceneConfig`` for ``formed-small-v1`` or ``formed-z1-v1``.
``generate_formed_scene(codebook, config=..., metadata=...)`` returns
``FormedScene`` with ordered ZYXC images, typed formed/per-round tables and
float64 NCR intended/pre-mix/realized signal tensors. The tensor's N axis is
explicitly labeled by ``amplicon_ids``. Tables join on namespace and amplicon ID;
they are distinct from detected candidate identities. Zero-brightness,
coincident and outside objects remain in truth. Visibility is recorded without
an eligibility policy. Unknown calibration stays unknown; supplied
``ImageMetadata`` is validated and preserved. Lengths are voxel indices even
when physical metadata is supplied.

Choose exactly one of coordinates/count/density; all omitted means count=8.
Explicit coordinates are in the supplied ID order and may be outside the grid;
singleton Z requires z=0. Gene IDs and supplied scalar properties are dictionaries
keyed by amplicon ID. Generated IDs are ``amplicon-0``, etc. Uniform, weighted
voxel-cell and rejected-normal cluster placement follow the specification.
Abundances follow saved codebook row order. Density draws a Poisson count;
``max_count`` overflow fails without truncating or retrying. This bounded API
requires max_count <=1024, ZYX <=(32,64,64) and 1–4 rounds with four channels.

``ScalarDistribution`` supports constant, uniform, lognormal, folded_lognormal
(elongation only), and ID-keyed supplied modes. Parameters and their domains are
documented in the specification and class. All-zero abundance/placement weights,
negative count/density/brightness, zero width, elongation below one, nonfinite
values, malformed codebooks/labels/spacing and incompatible distribution modes
fail explicitly. Counts/seeds reject Booleans. Unknown constructor fields fail;
unsupported effects cannot silently become active. Parameters are checked even
for an empty population. Extreme draws, placement rejection exhaustion and output
overflow fail rather than returning nonfinite images.

Rendering uses the specified ellipsoidal four-sigma support, float64 accumulation
in sorted amplicon-ID order, then one cast. Integers use nearest-even rounding and
saturation with clipping counts; floats retain their values without normalization.
All round effects, backgrounds, noise and geometry are disabled in this slice.
Intended, pre-mix and realized amplitudes are equal but independent arrays.
The clean model's identity transforms and complete per-round state are explicit.

SHA-256/PCG64 streams use stable component/entity names, never Python ``hash()``.
Actual random streams, descriptors, NumPy version, effective/requested config,
encoding/mapping, image hashes and geometry are retained in
``provenance["extensions"]["starfinder.synthetic"]``. Constant/supplied properties
consume no streams. Codebook/config identity is hashed; publishers must additionally
pin the generator code revision or source patch in their external artifact manifest.
This payload is an in-memory synthetic source extension, not a saved checkpoint
or a complete processing run manifest. Pass ``sources=[scene.provenance]`` to
``starfinder.provenance.RunRecorder`` to retain its source ID, catalog/selection,
explicit unsaved-source checksum reason and extension in a processing run.
Only development generation is supported.

Run the bounded example from ``src/python``::

   uv run python ../../docs/examples/formed_scene.py

It asserts independent 3D/Z=1 extraction and decoding values with distinct
candidate identities, then prints both preset provenance payloads. It writes
no files and makes no detection-accuracy or storage-round-trip claim.

.. literalinclude:: ../examples/formed_scene.py
   :language: python

Historical scene and result contracts
-------------------------------------

``render_spots(shape, spots, ...)`` accepts a pandas scene table with unique,
non-null ``spot_id`` and finite ``z, y, x, intensity, sigma`` columns. Coordinates
are zero-based ZYX voxel indices; sigma is positive. This replaces the old
integer tuple alias. Integer scene centers retain the historical Gaussian
kernel, float32 accumulation, background/noise draws and clipping/truncating
uint8/uint16 cast. Fractional centers are evaluated analytically on integer
voxels. ``generate_volume`` returns a uint8 ZYX array using this same renderer.
``get_preset_config(name).shape_zyx`` replaces the separate size lookup.

``generate_dataset(config=...)`` returns ``SyntheticDataset``. Its ordered
``rounds[fov_id][round_label]`` arrays are ZYXC, with matching ``metadata`` and
``channel_labels``. ``generate_registration_pairs(["tiny"])`` returns a mapping
of preset names to results whose rounds contain reference and moving ZYX images.
Registration preset shift ranges intentionally differ from sequencing presets.
Select registration presets explicitly; generation does not select experiments.

``spot_truth`` retains identities scoped by FOV/preset, round/channel labels,
rendered centers, continuous pre-rounding displaced centers where available,
intensity/sigma, rendered/eligible flags, exclusion reasons, frame and units.
Eligibility here means a center survived rendering bounds, not molecular or
scientific qualification. Reference truth uses the reference frame. Frames are
explicit; physical spacing/origin/direction remain unknown. Per-round shifts
and forward displacement fields are in ``perturbations``; fields are sampled
at integer scene centers and then rounded with nearest-even ties. These are
forward scene perturbations, not inverse registration pull transforms. No
implicit inversion is performed.

``molecular_truth`` is None. ``historical_truth`` retains old generator records,
including the v2 sequencing schema; field filenames there are serialization
names, not evidence that any files were written. ``config`` is an independent
copy and ``provenance`` records seeds, encoding and limitations. Barcode reversal
uses the shared ``EncodingConfig(reverse_bases=True).encode(barcode)``.

Scientific limitations
----------------------

`W-93 <https://linear.app/jiahaoh/issue/W-93>`_ owns qualification of shared
molecular truth and calibration. Registration seeds still use
``seed + hash(preset_or_deformation_name) % 10000``; resolved seeds are recorded,
but this is not cross-process reproducibility. Existing spot dropout,
integer rounding, per-round intensity/PSF jitter and random background are
preserved. ``background_std`` remains unused; even ``add_noise=False`` retains
the historical randomized background. No acquisition calibration or molecular
truth is invented by the namespace migration.

Example
-------

.. code-block:: python

   from starfinder.synthetic import SyntheticConfig, generate_dataset

   result = generate_dataset(SyntheticConfig(
       n_z=8, height=32, width=32, n_fovs=1, n_spots_per_fov=5,
   ))
   image = result.rounds["FOV_001"]["round1"]
   visible = result.spot_truth.loc[result.spot_truth.rendered]
