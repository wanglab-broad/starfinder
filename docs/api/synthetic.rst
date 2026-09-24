starfinder.synthetic
====================

Pure processed-image scene generation. Importing this package does not import
benchmark orchestration. Generation returns arrays in memory; the benchmark
persistence adapter writes workflow TIFFs, JSON, scene tables and annotations.

.. currentmodule:: starfinder.synthetic

.. autosummary::
   :toctree: generated

   generate_codebook
   generate_dataset
   generate_displacement_field
   generate_registration_pairs
   generate_volume
   get_preset_config
   render_spots
   SyntheticConfig
   SyntheticDataset

Scene and result contracts
--------------------------

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

Qualification of shared molecular truth and calibration remains open.
Registration seeds still use
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
