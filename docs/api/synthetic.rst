starfinder.synthetic
====================

One processed-image scene generator for oracle fixtures and benchmark data.
Importing this package does not import benchmark orchestration.
``generate_formed_scene`` implements the
:doc:`formed-amplicon model <../synthetic-specification>`; ``generate_dataset``
and ``generate_registration_pair`` build multi-FOV datasets and registration
pairs from the same scenes, and the benchmark presets below give them
documented appearance defaults. Generation returns arrays in memory or hands
each round to a callback; ``save_formed_scene`` writes formed-scene fixtures
and ``starfinder synthetic generate`` writes the MATLAB-compatible layout.

.. currentmodule:: starfinder.synthetic

.. autosummary::
   :toctree: generated

   BackgroundConfig
   BENCHMARK_PRESETS
   benchmark_scene_preset
   deformation_geometry
   DEFORMATION_PRESETS
   DEVELOPMENT_FACTORS
   DEVELOPMENT_FIXTURES
   development_preset_factors
   development_scene_preset
   DEVELOPMENT_SIZES
   formed_scene_preset
   FormedScene
   FormedSceneConfig
   forward_displacement
   generate_codebook
   generate_dataset
   generate_formed_scene
   generate_registration_pair
   GeometryConfig
   NoiseConfig
   PRESET_VERSION
   ReadoutEffectsConfig
   registration_scene_preset
   save_formed_scene
   ScalarDistribution
   SCENE_PRESETS
   SyntheticDataset
   TextureConfig

Formed scenes
-------------

``formed_scene_preset(name)`` returns a supplied ``Codebook`` and
``FormedSceneConfig`` for any ``SCENE_PRESETS`` name: the fixture tier
``formed-small-v1`` (8×32×32) or ``formed-z1-v1`` (1×32×32), the four
development fixtures, or one FOV of a benchmark preset. ``generate_formed_scene(codebook, config=..., metadata=...)`` returns
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

Rendering uses the ellipsoidal four-sigma support in sorted amplicon-ID order,
one channel plane at a time, then one cast. ``accumulation`` selects the
rendering dtype: ``None`` (default) means float64 for float64 output and
float32 otherwise; the fixture and development presets request float64 so
the independent oracle holds. Integers use nearest-even rounding and
saturation with clipping counts; a ``RuntimeWarning`` names the round when more
than 1% of its voxel values clip. Floats retain their values without
normalization. Rounds are generated one at a time; ``generate_formed_scene``
returns all of them, while ``generate_dataset`` can hand each round to a
callback so only one round is held in memory.
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
* ``dependent_enabled`` applies the signal-dependent residual to pre-noise total
  J: with ``model="gaussian"`` (default) it adds sqrt(alpha*J)*Z_dep; with
  ``model="poisson"`` it replaces J by alpha*Poisson(J/alpha), with the same
  mean and variance (alpha is intensity per detected count; zero draws
  nothing). Then ``independent_enabled`` adds read noise sigma*Z_ind.
  Nonnegative scalar ``alpha`` and ``sigma`` default zero. Both draw from the
  round/channel keyed noise streams in flat chunks, equal to one full-plane
  draw. Float output retains negative residuals; integer output
  rounds/saturates only once.

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

Alternatively ``local_magnitude`` gives each control a uniformly random
direction of that length.

Affine and polynomial terms add ``A(q-c)`` and ``P m(u)`` to the displacement,
with grid centre ``c=(shape-1)/2`` and ``u=(q-c)/h``, where the isotropic scale
``h`` is the largest grid half extent (at least 1). ``affine_zyx`` is R×3×3 in
voxels per voxel; ``polynomial_zyx`` is R×3×6 in voxels for the monomials
``zz, yy, xx, zy, zx, yx`` of u. Alternatively ``affine_max_zyx`` and
``polynomial_max_zyx`` draw uniform coefficients scaled so each output axis
displacement is at most that bound on the grid (exact at a corner for the
affine term). ``reference_round`` holds one round at identity and draws nothing
for it.

``translation_enabled``, ``local_enabled``, ``affine_enabled`` and
``polynomial_enabled`` default false. Disabled requests are retained and
validated; effective maps are identity. For Z=1, requested Z translations,
vectors, control-center Z and Z output rows must be zero. Each realized map
must satisfy ``sum(norm(v_k)/(l_k*sqrt(e))) + ||A||_F + J_P <= 0.5``, where J_P
bounds the polynomial Jacobian on the grid; violations error without
rescaling or redrawing. Actual inverse iterations, update, residual,
coefficients, frames and direction are recorded for each round in
``provenance["transforms"]``. A transform, its destination frame and the
effective geometry ``kind`` are ``identity`` exactly when the effective
coefficients are zero, whatever the enable flags say. No registration estimator
supplies truth; negating a nonlinear forward field does not invert it.

Molecules retain reference coordinates in ``formed`` and realized coordinates
in ``round_truth``. Kernels keep their widths and angle (no Jacobian shape
deformation). Backgrounds evaluate analytic ``B(F_inverse(p))`` on the output
grid; the inverse is evaluated only when a background component exists,
per axis for pure translations and otherwise in blocks of at most 2^18
voxels. With float32 accumulation (the benchmark tier) each point stops once
its own update is at most 1e-10 voxels; float64 accumulation (the fixture
tiers) iterates whole blocks. Both use the same tolerance and residual limit,
and ``inverse["stopping"]`` records which rule was used. Without a background the recorded inverse diagnostics check the
moved amplicon centers instead. ``forward_displacement(transform, shape)``
evaluates a recorded map as a float32 ``F(q)-q`` field, optionally for a
range of Z planes. ``scene.metadata`` is the reference grid; ``scene.round_metadata``
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

Benchmark datasets
------------------

``generate_dataset(codebook, config, fov_ids=("FOV_001",), preset=None,
on_round=None)`` generates one formed scene per FOV ID. Each FOV sets
``FOV_id`` and its own stream namespace, scene key ``[config.scene_key, FOV]``,
so appending IDs never changes earlier FOVs. With ``on_round(fov, round_label,
image, metadata)`` each round image is handed over when generated and not
retained. The result is ``SyntheticDataset``: ``rounds``/``metadata`` per FOV,
concatenated ``formed`` and ``round_truth`` (the namespace names the FOV),
``spot_truth`` (one row per FOV, amplicon and round with the codeword channel,
realized amplitude, visibility and a reason when not visible), per-FOV
``provenance`` and ``historical_truth``. The last is the ``ground_truth.json``
payload in the historical v2 keys (``image_shape``, ``n_rounds``, ``fovs`` with
``shifts`` and ``spots`` holding ``id``, ``gene``, ``barcode``, ``color_seq``,
reference ``position`` and ``intensity``), derived from the truth tables.
Positions and shifts are continuous voxel indices; no molecular truth is implied.

``generate_registration_pair(preset, deformation="shift", seed=None,
dtype="uint16", noise=True, include_reference=True, on_round=None)`` images
one scene in rounds ``reference`` and ``<deformation>``; only the moving round
moves. All pairs of a preset share the scene, so reference images and amplicons
are identical and ``include_reference=False`` skips rendering the reference
again. ``historical_truth["pairs"]`` gives the shift or the deformation kind and
Lipschitz bound; the transform records hold the full forward map.

``deformation_geometry(name, shape)`` maps the historical deformation names in
``DEFORMATION_PRESETS`` onto ``GeometryConfig``: YX magnitudes are a percent of
min(Y, X) capped in pixels, Z magnitudes a percent of Z. ``polynomial_*`` and
``linear_small`` use bounded polynomial or affine terms; ``gaussian_*`` and
``multi_point`` use RBF controls at fixed fractions of the grid with random
directions, reduced when needed to meet the invertibility bound. Random
polynomial and affine draws are reduced the same way when their Jacobian bound
would exceed it. ``gaussian_*`` keep their requested YX magnitude on every
benchmark preset; ``multi_point`` (four overlapping controls) is reduced below
``tissue``:

.. list-table:: Effective ``multi_point`` YX magnitude (requested → used, pixels)
   :header-rows: 1

   * - Preset
     - Requested
     - Used
   * - ``tiny``
     - 5.12
     - 1.32
   * - ``small``
     - 10.24
     - 2.64
   * - ``medium``
     - 20
     - 5.27
   * - ``large``, ``thick_medium``
     - 20
     - 10.54
   * - ``tissue``
     - 20
     - 20

The used magnitude is ``GeometryConfig.local_magnitude`` of
``deformation_geometry(name, shape)`` and is recorded with each transform.

``starfinder synthetic generate --mode e2e|registration --preset NAME --seed N
--owner NAME --output NEW_DIR [--dtype uint8|uint16] [--no-noise]`` writes rounds
as they are generated. E2E mode writes ``FOV_###/round#/ch##.tif`` (ZYX),
``codebook.csv`` (gene,barcode), ``ground_truth.json``, ``scene_truth.csv``,
``formed.csv``, ``round_truth.csv``, ``generation.json`` and ``manifest.json``.
Registration mode writes ``synthetic/<preset>/ref.tif``, ``mov_shift.tif``,
``mov_deform_<name>.tif`` (ch00, ZYX), ``field_<name>.npy`` (float32 Z×Y×X×3
forward displacement on the reference grid) and the same truth files, plus
``synthetic/summary.json``. ``generation.json`` keeps each scene's provenance
but replaces the ``stream_scheme`` descriptor list with ``stream_count`` and
``streams_per_component``: descriptors follow from the key scheme, and a
tissue-sized FOV draws about 10^5 of them.

Choosing a preset
-----------------

``SCENE_PRESETS`` is the single registry. Use the fixture and development tiers
(float64 accumulation, literal or independent-oracle checks) for tests of the
model itself, and the benchmark tier for pipeline, registration and timing
work. Benchmark presets keep the historical shapes, amplicon counts per FOV
(two FOVs), seeds and shift half-ranges:

.. list-table::
   :header-rows: 1

   * - Preset
     - ZYX shape
     - Amplicons/FOV
     - Seed
     - E2E shift z, yx
     - Registration shift z, yx
     - Genes
     - Use
   * - ``tiny``
     - 8×128×128
     - 10
     - 42
     - 2, 5
     - 2, 10
     - 12
     - examples, smoke tests
   * - ``small``
     - 16×256×256
     - 50
     - 42
     - 2, 5
     - 4, 25
     - 12
     - the test suite's session data
   * - ``medium``
     - 32×512×512
     - 400
     - 42
     - 8, 50
     - 8, 50
     - 12
     - development benchmarks (about 26 s e2e and 57 s registration on one thread)
   * - ``large``
     - 30×1024×1024
     - 1500
     - 123
     - 7, 100
     - 7, 100
     - 64
     - FOV-scale benchmarks
   * - ``tissue``
     - 30×3072×3072
     - 14000
     - 456
     - 7, 300
     - 7, 300
     - 64
     - tissue-scale memory and runtime
   * - ``thick_medium``
     - 100×1024×1024
     - 5200
     - 789
     - 25, 100
     - 25, 100
     - 64
     - thick sections

Appearance defaults (``PRESET_VERSION = "benchmark-presets-v1"``) replace the
historical constant background 20, noise σ 10 and uint8 amplitudes 200–255.
They are documented engineering choices for realistic-looking development
data, not calibration:

* uint16 output, float32 accumulation; ``dtype="uint8"`` scales every intensity
  by 1/16 so typical amplicons stay below 255.
* Amplitude lognormal with median 1500 and log-SD 0.4 (a brightness spread of
  roughly 0.5–2.2× across the central 95%). Gaussian widths lognormal with
  medians 1.5 (axial) and 1.3 (lateral) voxels and log-SD 0.1; mild elongation
  (folded lognormal, log-SD 0.1) at uniform angles.
* A camera offset of 100 per channel, and a spatially varying tissue background:
  a lateral gradient from 0.5 to 1.5 plus three broad Gaussian regions (heights
  1, 0.8, 0.6), weighted 40, 30, 30 and 20 per channel. It moves with the tissue.
* Poisson noise with alpha 1 plus Gaussian read noise σ 3.
* 5% crosstalk from each channel into the next and a per-round trend of 0.95.
* Round 1 is the reference; later rounds get uniform translations within the
  e2e half-range. Registration pairs use the registration half-range or a
  deformation preset.
* tiny/small/medium use the historical GeneA–GeneH plus GeneI–GeneL, so every
  channel holds amplicons in every round; larger presets use the 64-gene
  ``generate_codebook(64)``.

``SCENE_PRESETS[name]["peak_bytes_estimate"]`` estimates generation working
memory for one round (output round, the writer's contiguous copy of one
channel, two float planes and bounded blocks): about 0.30 GiB for ``medium``,
0.72 GiB for ``large``, 5.3 GiB for ``tissue`` and 2.1 GiB for ``thick_medium``,
plus the interpreter. ``medium`` generation is
tested with ``/usr/bin/time -v``; ``large``, ``tissue`` and ``thick_medium`` are
validated by configuration and this estimate only.

Scientific limitations
----------------------

Qualification of shared molecular truth and calibration remains open. Appearance
defaults are uncalibrated; no acquisition physics beyond the stated model is
claimed. All draws use the documented SHA-256/PCG64 stream keys, so generation
is byte-repeatable across processes for the same NumPy build on the same CPU.
NumPy selects CPU-specific implementations of transcendental functions such as
``exp``, so values derived from them (lognormal properties, kernels, backgrounds)
can differ by one unit in the last place between hosts; uniform draws and
positions do not.

Example
-------

.. code-block:: python

   from dataclasses import replace
   from starfinder.synthetic import benchmark_scene_preset, generate_dataset

   codebook, config = benchmark_scene_preset("tiny")
   result = generate_dataset(codebook, replace(config, shape_zyx=(8, 64, 64)),
                             fov_ids=("FOV_001", "FOV_002"), preset="tiny")
   image = result.rounds["FOV_001"]["round1"]          # ZYXC uint16
   visible = result.spot_truth.loc[result.spot_truth.rendered]
