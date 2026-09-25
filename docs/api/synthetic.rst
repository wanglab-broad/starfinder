starfinder.synthetic
====================

Pure processed-image scene generation. Importing this package does not import
benchmark orchestration. ``generate_formed_scene`` implements the
:doc:`formed-amplicon model <../synthetic-specification>`; the historical
generators keep their original semantics. Generation returns arrays in memory;
``save_formed_scene`` writes formed-scene fixtures, and the benchmark persistence
adapter writes historical workflow TIFFs, JSON, scene tables and annotations.

.. currentmodule:: starfinder.synthetic

.. autosummary::
   :toctree: generated

   BackgroundConfig
   DEVELOPMENT_FACTORS
   DEVELOPMENT_FIXTURES
   development_preset_factors
   development_scene_preset
   DEVELOPMENT_SIZES
   formed_scene_preset
   FormedScene
   FormedSceneConfig
   generate_codebook
   generate_dataset
   generate_displacement_field
   generate_formed_scene
   generate_registration_pairs
   generate_volume
   GeometryConfig
   get_preset_config
   NoiseConfig
   ReadoutEffectsConfig
   render_spots
   save_formed_scene
   ScalarDistribution
   SyntheticConfig
   SyntheticDataset
   TextureConfig

Formed scenes
-------------

``formed_scene_preset(name)`` returns a supplied ``Codebook`` and
``FormedSceneConfig`` for ``formed-small-v1`` (8×32×32) or ``formed-z1-v1``
(1×32×32). ``generate_formed_scene(codebook, config=..., metadata=...)`` returns
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
voxel-cell and rejected-normal cluster placement follow the model page.
Abundances follow saved codebook row order. Density draws a Poisson count.
The library has no upper bound on shape, round count or amplicon count; callers
own memory and time limits. ``max_count`` (default 1024) is a user-settable
guard: a larger N fails without truncating or retrying.

``ScalarDistribution`` supports constant, uniform, lognormal, folded_lognormal
(elongation only), and ID-keyed supplied modes. Angles are in [0, π): constant
and supplied values are checked against that domain, and uniform angles require
exactly (0, π). All-zero abundance/placement weights, negative
count/density/brightness, zero width, elongation below one, nonfinite values,
malformed codebooks/labels/spacing and incompatible distribution modes fail
explicitly. Counts/seeds reject Booleans. Parameters are checked even for an
empty population. Extreme draws, placement rejection exhaustion and output
overflow fail rather than returning nonfinite images.

Rendering uses the ellipsoidal four-sigma support, float64 accumulation in
sorted amplicon-ID order, then one cast. Integers use nearest-even rounding and
saturation with clipping counts; floats retain their values without normalization.
Readout effects, backgrounds, noise and geometry all default disabled. With
readout effects disabled, intended, pre-mix and realized amplitudes are equal
but independent arrays.

``scene.provenance`` is a JSON-serializable dict with exactly: ``generator`` and
``generator_version``; ``requested_config`` and ``effective_config`` (the latter
includes drawn counts, effective readout/background/noise values, background
components and the geometry ``kind`` derived from the effective transforms);
``seed`` and ``stream_scheme`` (key fields, namespace, hashing, bit generator,
NumPy version and the descriptors actually drawn); ``codebook``; per-round
``image_sha256``; ``clipping_counts``; and ``transforms``. Constant/supplied
properties and disabled controls draw nothing.

Configuration dataclasses are frozen but deliberately not hashable, because
several fields accept dicts or arrays; ``hash(config)`` raises ``TypeError``.
Use ``provenance["requested_config"]`` when a stable identity is needed.

Run the example from ``src/python``::

   uv run python ../../docs/examples/formed_scene.py

It asserts independent 3D/Z=1 extraction and decoding values with distinct
candidate identities, then prints both preset provenance payloads. It writes
no files and makes no detection-accuracy claim.

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
Wrong shapes, nonfinite/negative values, probabilities/factors above one and
overflow fail explicitly. A zero mixing matrix may suppress all emission
without marking a molecule lost. The order is survival/dropout, weakening,
trend, source gain, then mixing. Intended amplitudes and codewords never
change. Lost and invisible objects remain in both truth tables. These are
processed-image development controls, not calibrated chemical rates.

From ``src/python``::

   uv run python ../../docs/examples/readout_effects.py

The example checks exact histories [8,4,2], [8,0,2], [8,1,2] and [8,0,0],
asymmetric mixing [8,2,0,0], combined gain/trend/weakening/mixing, and center
extraction in 3D and Z=1. Detector ``spot-1`` is explicitly placed at ``gt-A``;
this is a supplied-coordinate extraction oracle, not detection accuracy.

.. literalinclude:: ../examples/readout_effects.py
   :language: python

Structured backgrounds and residual noise
-----------------------------------------

Pass ``background=BackgroundConfig(...)`` and ``noise=NoiseConfig(...)`` to
``FormedSceneConfig``. The defaults contribute exactly zero. Each Boolean enable
flag is independent; retained disabled parameters are validated but neither
rendered nor randomly sampled.

* ``baseline_enabled`` adds nonnegative ``baseline`` (R,C) in destination channel
  order, after signal/tissue assembly. None means zero; no broadcasting.
* ``gradient_enabled`` adds max(0, intercept + slopes dot normalized ZYX) to
  the reference scalar background. ``gradient_intercept`` is nonnegative;
  ``gradient_slopes_zyx`` may be signed. Singleton normalized coordinates are zero.
* ``regions_enabled`` adds untruncated Gaussian bumps with supplied
  ``region_centers`` (K,3), ``region_sigma_zyx`` (K,3) and ``region_heights`` (K,).
  None widths/heights select (2,8,8) and 10 per center; empty centers give zero.
* ``texture_enabled`` uses ``TextureConfig`` for count, Poisson density, or
  explicit coordinates, with the formed placement laws under separate background
  streams. Default enabled count is four, widths (1,3,3), peak height 5. Axial
  and lateral widths and height accept ``ScalarDistribution``; supplied values
  use ``blob-0``, etc. ``TextureConfig.max_count`` is the same kind of
  user-settable guard as the formed one.
* Nonnegative ``tissue_weights`` (R,C), default zero, scale the summed scalar
  background into destination channels. Neither mixing nor molecular loss acts
  on tissue or baseline. Background components are not cells or molecular truth.
* ``dependent_enabled`` adds sqrt(alpha*J)*Z_dep to pre-noise total J; then
  ``independent_enabled`` adds sigma*Z_ind. Nonnegative scalar ``alpha`` and
  ``sigma`` default zero. Float output retains negative residuals; integer
  output rounds/saturates only once.

``effective_config["background"]["components"]`` retains the analytic
reference-frame components, blob IDs, centers, widths and heights. Noise
strengths do not resample molecules, codewords, masks, latents or transforms.
These are effective processed-image approximations, not empirical tissue,
photon statistics or detector calibration. From ``src/python``::

   uv run python ../../docs/examples/background_noise.py

The example checks gradient/Gaussian literals, infinite tails and noise
isolation in (3,7,9) and (1,7,9), with no saved images.

.. literalinclude:: ../examples/background_noise.py
   :language: python

Shared analytic geometry
------------------------

Pass ``geometry=GeometryConfig(...)`` to ``FormedSceneConfig``. It uses
``F(q) = q + sum(v_k * exp(-||q-c_k||²/(2*l_k²))) + t`` in reference ZYX voxel
indices. Local displacement is evaluated before translation, always at the
reference point; rounds never accumulate motion. ``translations_zyx`` is R×3,
``centers_zyx`` K×3, ``scales`` K (default 8), and ``vectors_zyx`` R×K×3.
Alternatively use uniform ``translation_max_zyx`` half-ranges or normal vector
component SD ``strength``. Supplied coefficients consume no random draws.

``translation_enabled`` and ``local_enabled`` default false. Disabled requests
are retained and validated; effective maps are identity. For Z=1, requested Z
translations/vectors and control-center Z must be zero. Each realized map must
satisfy ``sum(norm(v_k)/(l_k*sqrt(e))) <= 0.5``; violations error without
rescaling or redrawing. Actual inverse iterations, update, residual,
coefficients, frames and direction are recorded for each round in
``provenance["transforms"]``. A transform, its destination frame and the
effective geometry ``kind`` are ``identity`` exactly when the effective
coefficients are zero, whatever the enable flags say. No registration estimator
supplies truth; negating a nonlinear forward field does not invert it.

Molecules retain reference coordinates in ``formed`` and realized coordinates
in ``round_truth``. Kernels keep their widths and angle (no Jacobian shape
deformation). Backgrounds evaluate analytic ``B(F_inverse(p))`` on the output
grid. ``scene.metadata`` is the reference grid; ``scene.round_metadata``
supplies each output's destination frame with the same physical calibration.

From ``src/python``::

   uv run python ../../docs/examples/formed_geometry.py

The example keeps a shared molecular/background landmark, a nonconstant local
map followed by translation, and an out-of-frame round in 3D and Z=1.

Development fixtures
--------------------

``development_scene_preset(condition="clean", size="small")`` returns the
``controlled-development-v1`` codebook/config described on the
:doc:`model page <../synthetic-specification>`. ``DEVELOPMENT_SIZES`` holds
``z1`` and ``small``; ``DEVELOPMENT_FIXTURES`` maps the four packaged fixture
names (``z1-clean``, ``z1-combined``, ``small-clean``, ``small-combined``) to
their condition and size. One fixture amplicon is elongated and rotated
(θ = π/6). The tests save each fixture and compare it with an independent
oracle. Other conditions in ``DEVELOPMENT_FACTORS`` produce on-demand
single-factor comparisons; they are neither packaged nor oracle-checked.

``save_formed_scene(scene, directory)`` writes ``images/<round>.tif`` with
:py:func:`starfinder.io.save_volume` (ZYXC, round metadata), ``formed.csv``,
``round_truth.csv``, ``signals.csv`` and ``provenance.json`` into a new or
empty directory.

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

Historical scientific limitations
---------------------------------

Qualification of shared molecular truth and calibration remains open.
Registration seeds still use
``seed + hash(preset_or_deformation_name) % 10000``; resolved seeds are recorded,
but this is not cross-process reproducibility. Existing spot dropout,
integer rounding, per-round intensity/PSF jitter and random background are
preserved. ``background_std`` remains unused; even ``add_noise=False`` retains
the historical randomized background. No acquisition calibration or molecular
truth is invented by the namespace migration.

Historical example
------------------

.. code-block:: python

   from starfinder.synthetic import SyntheticConfig, generate_dataset

   result = generate_dataset(SyntheticConfig(
       n_z=8, height=32, width=32, n_fovs=1, n_spots_per_fov=5,
   ))
   image = result.rounds["FOV_001"]["round1"]
   visible = result.spot_truth.loc[result.spot_truth.rendered]
