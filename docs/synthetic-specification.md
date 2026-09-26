# Formed-amplicon synthetic model

This page defines the model behind `starfinder.synthetic.generate_formed_scene`:
its stages, the order of the effects and the random streams. The
{doc}`synthetic API page <api/synthetic>` documents the Python interface,
including the multi-FOV datasets, registration pairs and benchmark presets
built from these scenes.

## Scope

The scene starts **after amplicon formation**. Identity and reference position
belong to an amplicon; each round is a new readout of the same population, not a
new amplification or RNA draw. Stripping readout probes does not delete
amplicons, so temporary absence can recover in a later round, while persistent
loss cannot. The model produces processed images: Gaussian widths describe
effective puncta, not an optical point-spread function; noise is a residual
image model, not photon counts; background components are smooth mathematical
structures, not cells or measured tissue. Defaults are clean development
choices, not fitted ranges, and nothing on this page is a calibration claim.

## Inputs and conventions

* One scene is one dataset/sample/FOV with a validated `barcode.Codebook`
  (ordered round and channel labels, four channels, colors 1–4 mapped to
  channel indices through the supplied bijection), optional reference
  `ImageMetadata` and an unsigned 64-bit root seed (default 42).
* Images are ZYXC. Coordinates and widths are float64 ZYX voxel indices, also
  when physical metadata is supplied. Z=1 stays 3D and samples z=0; it is not a
  projection.
* Shape dimensions are positive integers; the round count comes from the
  codebook. The library sets no upper bound on shape, rounds or counts: callers
  own memory and time. `max_count` (default 1024) is a user-settable guard, so
  a sampled or explicit N above it fails rather than being truncated.
* Latents, truth and signal tensors are float64. Images accumulate in the
  `accumulation` dtype: float32 by default, float64 for float64 output or when
  requested (the fixture and development presets request it). Output is
  float32 by default (float64, uint8 and uint16 are explicit alternatives),
  cast once at the end; more than 1% clipped voxel values in a round warns. Integer output rounds
  to nearest even, then saturates and counts clipped voxels. Float output keeps
  negative residuals. Nothing is normalized.
* Every parameter must be finite. Booleans are rejected where integers are
  required. Unknown modes, duplicate IDs, wrong shapes, non-NFC labels and
  out-of-domain values raise before any output is produced.

## Stages and effect order

Every round is generated from the same reference scene. Transforms and trends
are absolute with respect to the reference; nothing accumulates across rounds.

1. **Formed population.** Exactly one of explicit coordinates, `count` or
   `density` selects N (default count 8). Density draws N ~ Poisson(density × Z×Y×X).
   Placement is uniform on [0, L−1) per nonsingleton axis, weighted (choose a
   voxel by normalized ZYX weights, then jitter uniformly inside its cell
   clipped to [0, L−1]) or clustered (choose a supplied center by weight, add
   normal offsets with `spread_zyx`, reject the whole proposal outside the
   grid, at most 10,000 proposals). Explicit coordinates may lie outside the
   grid and are kept. Genes are supplied per ID or drawn from abundance weights
   in codebook row order.
2. **Persistent properties.** Each amplicon keeps peak brightness A, axial
   width sz, lateral width sl, elongation e ≥ 1 and angle θ in [0, π) for all
   rounds. Laws are constant, uniform, lognormal (log median, log SD),
   folded lognormal (elongation only: exp(μ + |τZ|)) or supplied per ID; θ is
   constant, supplied or uniform on (0, π). Defaults are A=100, sz=sl=e=1, θ=0.
3. **Intended signal.** `intended[i, c, r] = A_i` for the channel encoded by the
   codeword in round r, else 0. Codewords are never rewritten.
4. **Readout effects**, in this order, each with its own enable flag:
   persistent loss (one draw per ID; lost from `loss_start` onward),
   temporary dropout (per ID and round), temporary weakening (per ID and round,
   multiplier `weak_factor[r]`), progressive trend (`b**r`, with `b**0 = 1`) and
   source-channel gains (R×C). The product is `pre_mix`.
5. **Spectral mixing.** `realized[d] = Σ_s M[r, d, s] · pre_mix[s]` with
   destination rows and source columns in codebook channel order. Mixing is not
   normalized and acts on amplicon signal only.
6. **Geometry.** Each round r maps reference points by
   `F_r(q) = q + d_r(q) + t_r`, with
   `d_r(q) = Σ_k v_rk · exp(−‖q − c_k‖² / (2 l_k²)) + A_r(q − c) + P_r m(u)`,
   grid centre `c = (shape − 1)/2`, `u = (q − c)/h` with `h` the largest half
   extent (at least 1) and `m(u) = (u_z², u_y², u_x², u_z u_y, u_z u_x, u_y u_x)`.
   Translations are supplied or drawn uniformly in [−a, a); local vectors are
   supplied, drawn with normal components of SD `strength`, or drawn as
   uniformly random directions of length `local_magnitude`; affine and
   polynomial coefficients are supplied or drawn uniformly and scaled to a
   per-axis displacement bound on the grid. A named reference round is held at
   identity. Each map must satisfy
   `Σ_k ‖v_rk‖ / (l_k √e) + ‖A_r‖_F + J_P ≤ 0.5` (J_P bounds the polynomial
   Jacobian on the grid), which makes the inverse fixed point
   `q ← p − t − d(q)` contractive (stop at update ≤ 1e−10 voxels within 100
   iterations; composition residual ≤ 2e−10). For Z=1, all requested Z motion,
   control-center Z and Z output rows must be zero. A round's transform is labeled
   `identity` exactly when its effective coefficients are zero.
7. **Rendering.** Molecules move with their centers and keep their widths and
   angle. At integer voxel p about the moved center, with `dz, dy, dx = p − q`,
   `u = cos θ·dy + sin θ·dx` and `v = −sin θ·dy + cos θ·dx`, the kernel is
   `exp(−½((dz/sz)² + (u/(sl·e))² + (v/sl)²))` inside the squared normalized
   radius 16 and exactly 0 outside. Kernels are summed in sorted amplicon-ID
   order and scaled by the realized amplitude.
8. **Tissue-like background.** A scalar field B, defined analytically in the
   reference frame and evaluated at `F_r⁻¹(p)`, is scaled into channels by
   nonnegative R×C `tissue_weights`. B sums an optional clamped linear gradient
   `max(0, a0 + a·q/(shape − 1))` (singleton axes use 0), optional supplied
   Gaussian regions and optional texture blobs placed like the formed
   population. Gaussian tails are untruncated; there is no periodic wrap. The
   inverse is evaluated only when B has a component: exactly per axis for a
   pure translation, otherwise block by block over the output grid (per-point
   stopping at the same tolerance for float32 accumulation).
9. **Baseline.** Nonnegative R×C offsets are added in the destination frame;
   they are not moved, mixed or lost.
10. **Noise.** With J the pre-noise total (signal, tissue and baseline), the
    dependent residual adds `sqrt(alpha·J)·Z_dep` (`model="gaussian"`) or
    replaces J by `alpha·Poisson(J/alpha)` (`model="poisson"`, same mean and
    variance); then read noise adds `sigma·Z_ind`. Draws come from separate
    streams per round and channel in C-order ZYX (drawn in flat chunks, which
    equals one full-plane draw).
11. **Cast** once to the output dtype, recording clipping counts.

All readout, background, noise and geometry controls default to disabled.
Disabled controls are still validated but become algebraic identities (zero,
one, identity matrix or identity map) and consume no random draws. The
requested and effective configurations are both recorded.

## Truth

* `formed` has one row per amplicon: namespace, amplicon ID, gene, codeword,
  reference frame, `formed_index`, reference z/y/x and A, sz, sl, e, theta.
* `round_truth` has N × R rows: moved z/y/x in the round's frame, transform ID,
  dropped/weakened/lost/emitting flags, `center_in_bounds` (closed bounds
  0..L−1), `support_intersects` (any voxel inside the kernel support),
  `support_truncated` (the axis-aligned extent 4·sz, 4·sl·√(e²cos²θ + sin²θ)
  and 4·sl·√(e²sin²θ + cos²θ) leaves the grid), nullable `first_loss_round`
  and the trend/weak multipliers.
* `intended`, `pre_mix` and `realized` are float64 N×C×R arrays labeled by
  `amplicon_ids`, `channel_labels` and `round_labels`.
* Lost, dark, overlapping and out-of-frame amplicons are kept. N=0 yields typed
  empty tables and (0, C, R) arrays. No eligibility or detectability is inferred.

## Random streams

Every random draw comes from its own generator, keyed by a JSON array in this
exact order:

`["starfinder.synthetic/1", split, seed, scene_key, component, entity, round_label, channel_label]`

* The first element is a fixed namespace string. `split` (default
  `development`) and `scene_key` are nonempty NFC labels; changing either
  redraws the scene. `seed` is an integer in [0, 2⁶⁴ − 1]. Unused fields are null.
* The array is serialized as compact UTF-8 JSON (`ensure_ascii=False`,
  separators `,` and `:`), hashed with SHA-256, and the whole digest, read as a
  big-endian unsigned integer, seeds NumPy `PCG64` wrapped in
  `numpy.random.Generator`. Python `hash()`, global generators and wall-clock
  seeds are never used, so results do not depend on `PYTHONHASHSEED`.
* Components: `count`, `placement`, `identity`, `brightness`, `width.axial`,
  `width.lateral`, `elongation`, `angle`, `round.dropout`, `round.weakening`,
  `round.loss`, `geometry.translation`, `geometry.local`, `geometry.affine`,
  `geometry.polynomial`, `background.count`,
  `background.placement`, `background.width`, `background.brightness`,
  `noise.dependent`, `noise.independent`.
* Persistent draws use the amplicon or blob ID as entity; per-round draws add
  the round label; noise uses round and channel. Texture widths use the entity
  `["blob-N","axial"]` or `["blob-N","lateral"]`; local vectors use
  `["control-N","vector"]` per round and draw Z, Y, X in order. Cluster choice
  and rejected proposals consume that amplicon's own placement stream.
* Consequences: changing one factor's parameters, appending IDs or reordering
  work never changes another component's draws. Multi-FOV datasets give each
  FOV the scene key `[scene_key, FOV]`, so adding FOVs changes no other FOV.
  Changing the shape can change the noise array layout. Repeatability is exact for a pinned NumPy build on the
  same CPU. NumPy releases do not promise identical distribution algorithms, and
  CPU-specific implementations of `exp` can change derived values by one unit in
  the last place between hosts.

## Development fixtures

`development_scene_preset(condition, size=...)` builds a two-amplicon scene
(`gt-A` at (z, 10, 10) with e=1, θ=0; `gt-B` at (z, 22, 22) with e=1.5, θ=π/6),
A=8, sz=1, sl=1.25, seed 42 and scene key `controlled-development-v1`, in
sizes `z1` (1×32×32) and `small` (9×32×32). Four fixtures are packaged:
`clean` and `combined` in each size. `combined` enables weakening (middle round
0.25), trend (base 0.5), gains (0.5), mixing, baseline, gradient, regions,
texture, both noise terms, translation and one local control, but not loss or
dropout. Tests save these fixtures with `save_formed_scene` (TIFF images, CSV
truth) and compare every reloaded voxel and truth value with an independent
oracle. Single-factor conditions remain available on demand; they are
comparisons against `clean`, not packaged or oracle-checked fixtures.
