starfinder.synthetic
====================

The :doc:`formed-amplicon specification <../synthetic-specification>` defines
``starfinder.synthetic/1`` and independent acceptance cases for the new
development model. ``generate_formed_scene`` implements its scene/truth and controlled readout
and background/noise slices; the historical APIs retain their original semantics.

Pure processed-image scene generation. Importing this package does not import
benchmark orchestration. Generation returns arrays in memory; the benchmark
persistence adapter writes workflow TIFFs, JSON, scene tables and annotations.

.. currentmodule:: starfinder.synthetic

.. autosummary::
   :toctree: generated

   BackgroundConfig
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
   NoiseConfig
   ReadoutEffectsConfig
   render_spots
   ScalarDistribution
   SyntheticConfig
   SyntheticDataset
   TextureConfig

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
Backgrounds and noise default disabled; geometry remains identity. With readout effects disabled,
intended, pre-mix and realized amplitudes are equal but independent arrays.
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

Controlled readout
------------------

Pass ``readout=ReadoutEffectsConfig(...)`` in ``FormedSceneConfig``. Each effect
has an independent Boolean ``*_enabled`` switch, default false. The generator
validates requested parameters even when disabled or N=0. It records both the
requested values and effective identities; disabled controls draw no randomness.

* ``dropout_probability`` and ``weakening_probability`` are length-R sequences
  in [0,1], default zero. ``weak_factor`` is length R in [0,1], default one.
  Enable with ``dropout_enabled`` and ``weakening_enabled``. Separate per-ID,
  round-label keyed uniform draws select each event; overlapping flags survive.
* ``trend_enabled`` applies ``trend_base**r`` with base in [0,1], default one,
  and zero-based acquisition index r. Even base zero gives multiplier one at r=0.
* ``loss_enabled`` selects IDs once using ``loss_probability`` in [0,1], default
  zero. ``loss_start`` is a zero-based acquisition index, default 1 when R>1,
  otherwise 0. Selected objects remain lost thereafter. Their nullable
  ``first_loss_round`` records this planned start even in preceding history rows.
* ``gain_enabled`` applies supplied nonnegative finite ``gains`` of shape (R,C)
  to source amplitudes, default ones.
* ``mixing_enabled`` applies supplied nonnegative finite ``mixing`` of shape
  (R,C,C), default per-round identity. Destination rows and source columns use
  the codebook's explicit channel order: ``realized[d] = sum_s M[d,s]*pre_mix[s]``.
  No normalization, implicit broadcasting, carryover or new identity is inferred.

``None`` selects defaults only for the sequences, matrices and ``loss_start``.
Loss/trend probabilities and scales are finite scalar numbers; flags and the
integer loss index are type-checked. Wrong shapes, nonfinite/negative values,
probabilities/factors above one and overflow fail explicitly. A zero mixing
matrix may suppress all emission without marking a molecule lost.

The order is survival/dropout, weakening, trend, source gain, then mixing.
Intended amplitudes and codewords never change. ``pre_mix`` and ``realized``
retain full NCR amplitude histories; ``round_truth`` retains each effect flag,
weak/trend multipliers, loss start and emission/geometry visibility separately.
Gains and matrices are shared across objects and retained in effective config.
Lost and invisible objects remain in both truth tables. Changing readout controls
preserves base positions, identities, widths and unrelated component draws.
The existing clean presets retain their numerical outputs; generator version 2
adds these controls and records the more detailed order/configuration.
These are processed-image development controls, not calibrated chemical rates.

From ``src/python``::

   uv run python ../../docs/examples/readout_effects.py

The example checks exact A4 histories [8,4,2], [8,0,2], [8,1,2] and [8,0,0],
asymmetric A5 mixing [8,2,0,0], combined gain/trend/weakening/mixing, and center
extraction in 3D and Z=1. Detector ``spot-1`` is explicitly placed at ``gt-A``;
this is a supplied-coordinate extraction oracle, not detection accuracy.
The background/noise example and tests cover destination baseline addition in A5.

.. literalinclude:: ../examples/readout_effects.py
   :language: python

Structured backgrounds and residual noise
-----------------------------------------

Pass ``background=BackgroundConfig(...)`` and ``noise=NoiseConfig(...)`` to
``FormedSceneConfig``. Generator version 3 adds these controls without changing
the clean preset image values. The defaults contribute exactly zero. Each
Boolean enable flag is independent; retained disabled parameters are validated
but neither rendered nor randomly sampled.

* ``baseline_enabled`` adds nonnegative ``baseline`` (R,C) in destination channel
  order, after signal/tissue assembly. None means zero; no broadcasting.
* ``gradient_enabled`` adds max(0, intercept + slopes dot normalized ZYX) to
  the reference scalar background. ``gradient_intercept`` is nonnegative;
  ``gradient_slopes_zyx`` may be signed. Singleton normalized coordinates are zero.
* ``regions_enabled`` adds untruncated Gaussian bumps with supplied
  ``region_centers`` (K,3), ``region_sigma_zyx`` (K,3) and ``region_heights`` (K,).
  None widths/heights select (2,8,8) and 10 per center; empty centers give zero.
* ``texture_enabled`` uses ``TextureConfig`` for count, Poisson density, or
  explicit coordinates. Uniform, weighted and clustered placement reuse the
  formed placement laws, under separate background streams. Default enabled
  count is four, widths (1,3,3), peak height 5. Axial and lateral widths and
  height accept ``ScalarDistribution``; lateral width is shared by Y and X.
  Supplied property values use ``blob-0``, etc. Width stream entities are compact
  JSON [blob ID, "axial"] and [blob ID, "lateral"]. Count over max_count fails
  without capping or retrying. All Gaussian tails are untruncated.
* Nonnegative ``tissue_weights`` (R,C), default zero, scale the summed scalar
  background into destination channels. Neither mixing nor molecular loss acts
  on tissue or baseline. Background components are not cells or molecular truth.
* ``dependent_enabled`` adds sqrt(alpha*J)*Z_dep to pre-noise total J; then
  ``independent_enabled`` adds sigma*Z_ind. Nonnegative scalar ``alpha`` and
  ``sigma`` default zero. J includes signal, tissue and baseline. Float output
  retains negative residuals; integer output rounds/saturates only once.

``background_components`` in the synthetic provenance extension retains
analytic reference-frame coefficients, blob IDs, centers, widths and heights.
The internal evaluator accepts real reference coordinates so the geometry
component can later supply inverse-mapped output positions. This release uses
identity geometry; no deformation has been implemented here. No existing noisy
image is sampled or given a second noise model. Requested configuration,
effective baselines/weights/noise, round/channel stream descriptors, standardized
draw hashes, pre-noise image hashes and final clipping counts remain separate.
Noise strengths do not resample molecules, codewords, masks, latents or transforms.

These are effective processed-image approximations, not empirical tissue,
photon statistics or detector calibration. From ``src/python``::

   uv run python ../../docs/examples/background_noise.py

The example checks gradient/Gaussian literals, infinite tails and noise isolation
in (3,7,9) and (1,7,9), with no saved images. Focused tests additionally check A5/A6,
independently derived PCG64 draws, residual mean/variance, invalid parameters,
zero/disabled controls and unchanged molecular truth. Run resource measurements
and exact outcomes belong in the issue's external manifest, not this API guide.

.. literalinclude:: ../examples/background_noise.py
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
