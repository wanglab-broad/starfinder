# Formed-amplicon synthetic specification

**Contract ID: `starfinder.synthetic/1`.** W-155 freezes the processed-image
model and independent acceptance cases below before dependent implementation.
This is a specification, not a claim that `SyntheticConfig` or the historical
generators implement it. Producers must record this ID and the accepted Git
revision or reviewed source snapshot hash. Changes to meanings require a new
version. The [artifact contract](artifact-contracts.md) is `starfinder.artifacts/1`,
accepted at `db3667bd9eef7ee6bdea6fe09fae8af3eb1158cc`; its E1 case applies.
The [array contracts](api/contracts.md) remain authoritative at processing boundaries.

## Assay rationale and limits

Ren et al., *Nature Protocols* 21, 1629–1661 (2026),
[doi:10.1038/s41596-025-01248-3](https://doi.org/10.1038/s41596-025-01248-3),
Figures 3–4 ([accessible captions](https://pubmed.ncbi.nlm.nih.gov/41028562/)),
describe amplification and hydrogel anchoring before repeated SEDAL readout,
imaging and stripping of fluorescent probes. We start **after formation**:
identity and reference position belong to an amplicon; a round is a new readout
of that population, not a new amplification or independent RNA draw. Stripping
fluorescent readout probes does not mean deleting every amplicon. Thus transient
absence can recover in a later round, while an explicitly selected persistent
loss cannot. This does not identify chemical loss rates or model reaction kinetics.

The [W-90](https://linear.app/jiahaoh/issue/W-90)/
[W-121](https://linear.app/jiahaoh/issue/W-121) simulator survey (September 17
attachment `ab6dd31d-a598-49f4-98b5-1b6105842303`, SHA-256
`49e7d06b44ab1671c3621aa3a99bb31edeb9b1fcb27c7e87ba6983ea137a86ae`)
distinguishes latent truth from observability and processing outputs. Its later
W-121 discussion selects processed images. The survey's older raw-optics pilot
is superseded by that decision and the current §2.12 scope.
[BarDensr](https://doi.org/10.1371/journal.pcbi.1008256) supplies precedent for
explicit codebook response, channel mixing, round/channel gains and structured
background. Its carryover model is not adopted here.
[Sim-FISH](https://sim-fish.readthedocs.io/en/latest/image_simulation.html) is a
precedent for Gaussian spot parameters and explicit positions; microsim's
[stages](https://talleylambert.com/microsim/stages/) motivate separating scene
from observation. No source licenses or algorithms are copied; no package is
selected for integration.

Gaussian widths below describe effective processed puncta, not an optical PSF.
Noise is a residual image model, not photon counts or detector read noise.
Background objects are mathematical structures, not cells or measured tissue.
Numerical defaults are clean software-development choices, never fitted ranges.
Original RNA abundance, amplification efficiency, carryover, raw acquisition,
restoration, cellular assignment and calibrated realism are outside this version.
W-93/W-124 must settle empirical ranges, independent experimental splits and
scientific tolerances; W-152/W-57 owns any changed scientific meaning. Those
questions block calibration claims, not these explicit development cases.

## Inputs, coordinates and validation

One scene describes one dataset/sample/FOV with ordered unique round and channel
labels, a supplied validated `barcode.Codebook`, reference `ImageMetadata`, and
an explicit unsigned 64-bit root seed (default 42). R is 1–4 and C is exactly 4
for the current color codebook. Color symbols 1–4 map through the supplied
bijection to channel indices; never sort labels or infer wavelength order.
Base encoding, reversal and split use the shared `EncodingConfig` once and are
saved with the encoded codebook. No implicit codebook generation is required.
Images are ZYXC; coordinates and widths are float64 ZYX voxel indices. Z=1 stays
3D, with formed z=0. Physical fields default to null, not fictitious micrometres.
Optional supplied calibration uses the existing image metadata conversion;
all model lengths remain explicitly voxel-index units in v1.

New scene shapes have positive integer dimensions at most (32,64,64). Default
shape is (8,32,32). Output defaults to float32; float64, uint8 and uint16 are
explicit alternatives. Compute in float64 and sum in stable amplicon-ID order;
cast once at the end. Float output retains negative residuals. Integer output
rounds nearest-even then saturates; record clipped voxel counts. No normalization.
All parameters must be finite; Booleans are not accepted as counts/seeds. Unknown
fields/modes, duplicate IDs, invalid labels/mappings, wrong ranks/lengths, negative
scales, zero widths and nonfinite generated values error before successful output.
Only development namespace generation is authorized; reserve calibration and
evaluation names without generating those scenes.

### Formed population (stage 1)

Exactly one of explicit coordinates, `count`, or `density` is selected. Default
is count=8, uniform placement. Explicit coordinates determine N and may be
outside the FOV (boundary probes); no clipping, deduplication or resampling.
Counts are nonnegative integers; N=0 is valid. Density is nonnegative formed
amplicons per voxel, with volume V=Z*Y*X even for Z=1. Draw N~Poisson(density*V)
from the count stream; density is an expected count, not a forced rounded count.
A declared `max_count` (default 1024) bounds allocation: explicit or sampled
N above it fails without capping or retrying. This is a resource guard, not
truncation of the Poisson distribution. Positive density with a Z=1 image does
not imply physical volumetric density. Count and density together always error.

| Placement | Exact law and parameters | Persistent output / clean behavior |
| --- | --- | --- |
| Uniform (default) | Independent U[0,L−1) on each nonsingleton axis; singleton=0 | One fractional reference position per ID; no minimum distance or overlap rejection |
| Spatial weights | Supplied finite nonnegative ZYX weights with positive sum; select voxel by normalized weights in C order, then uniform jitter inside its voxel cell intersected with [0,L−1] (singleton=0) | Fixed N; weights control relative placement, not additional abundance |
| Clustered | Supplied finite in-bounds K×3 centers, positive K weights and positive axial/lateral spread; select cluster categorically per ID, draw independent normal offsets, rejecting the whole position outside [0,L−1] | Default enabled-cluster spread (1,2,2) voxels; max 10,000 proposals/ID then explicit error; singleton offset=0; no edge clipping mass |
| Explicit | Supplied N×3 finite coordinates in stable ID order | No placement draw; identical/coincident and out-of-frame objects retained |

There is no hidden cell or cluster population. Cluster configuration must supply
centers; no guessed K. For weighted placement, boundary voxel cells are
[max(0,k−0.5),min(L−1,k+0.5)]; zero-width singleton cells yield 0.

Gene IDs are supplied per amplicon or drawn categorically in saved codebook row
order from finite nonnegative abundance weights (default all equal, positive
sum). Normalize once; zero weight means no random assignments of that gene.
Supplied IDs must occur in the codebook. These are **formed-amplicon** abundances;
no inference back to RNA copy number. A change of abundance never resamples
positions, widths or brightness for existing IDs.

### Brightness and shape (stage 2)

Each amplicon has a persistent peak brightness A, axial sigma sz, lateral sigma
sl, lateral elongation e and angle theta. Default A=100 intensity units,
sz=1, sl=1 voxels, e=1 and theta=0 radians. Axis widths are standard deviations,
not FWHM; FWHM=2*sqrt(2*ln(2))*sigma. Each scalar supports a constant, uniform
[a,b) (a=b is the constant), or lognormal exp(mu+tau*Z), Z~N(0,1), with tau>=0.
For lognormal specify **log median** mu and log standard deviation tau, not
arithmetic mean/SD. Brightness must be >=0, widths >0, elongation >=1.
Elongation lognormal requires mu>=0 and is exp(mu+abs(tau*Z)); this folded-log
law is named `folded_lognormal`, not ordinary lognormal. Theta supports constant
or U[0,pi), default constant 0. Invalid distribution/domain combinations error;
no post hoc absolute value/clipping for ordinary brightness or widths. Supplied
per-ID values override a distribution only through an explicit supplied mode.
Independent property streams prevent changing width distribution from redrawing A.

At integer sample p about transformed center q, set dz=pz−qz,
(u,v)=(cos(theta)*dy+sin(theta)*dx, −sin(theta)*dy+cos(theta)*dx).
The unit peak kernel is exp(−0.5*((dz/sz)^2+(u/(sl*e))^2+(v/sl)^2)).
Use the finite support ellipsoid with squared normalized radius <=16 (4 sigma);
outside support contributes exactly zero. Include every in-grid voxel inside
that support, even when its center is out of frame. Do not discard edge centers,
round fractional coordinates, renormalize truncated kernels or scale A by width.
Z=1 samples the same formula at z=0; it is not a projection. Shape properties
persist across rounds in v1; unconfigured per-round width jitter is forbidden.

## Round order and controlled effects

Process each ordered round against the same reference scene; transforms and
intensity trends are absolute relative to reference, never accidentally cumulative.
The first round's transform is identity in presets, but labels do not imply it.

1. Form population, assign codewords and persistent brightness/shape once.
2. For round r, derive intended channel amplitudes from the codeword and A.
3. Apply persistent survival, temporary dropout/weakening and progressive trend.
4. Apply source-channel round gains, then spectral mixing.
5. Transform centers and the persistent tissue-like background with the same
   forward map; evaluate spot kernels and background on the output grid.
6. Add destination-channel round baselines, then signal-dependent residual noise,
   then independent residual noise, finally cast/quantize once.
7. Record complete truth and visibility before downstream processing or QC.

| Effect | Input, default and law | Output, persistence and interaction |
| --- | --- | --- |
| Intended signal | A and encoded color c(i,r) | `intended[i,c,r]=A_i` for selected source channel, otherwise 0; never rewritten by loss/mixing |
| Temporary dropout | p_drop[r] in [0,1], default 0; U<p independently per ID/round | Boolean dropped; zero emission for this round only; next round can recover |
| Temporary weakening | p_weak[r] in [0,1], default 0; factor w[r] in [0,1], default 1; separate U<p | Multiplier w or 1 for this round; retain flag even if also dropped/lost |
| Progressive brightness | b in [0,1], default 1 | Multiplier b**r (zero-based acquisition index), with b**0=1 including b=0; cumulative effective trend, not a kinetic rate |
| Persistent loss | p_loss in [0,1], default 0; `loss_start` integer r, default 1 when R>1 else 0; U<p once per ID | Selected IDs absent at all r>=loss_start; null first_loss for survivors; no recovery or truth deletion |
| Round/channel gain | Explicit nonnegative R×C array g, default ones | Scales source-channel signal after above multipliers; no random hidden jitter |
| Mixing | Explicit nonnegative C×C M per round, default identity | `realized[d]=sum_s M[d,s]*pre_mix[s]`; destination rows/source columns; no normalization, diagonals may attenuate; applies to amplicon signal only |
| Baseline | Nonnegative R×C beta, default zero, intensity units | Added in destination image frame after signal/tissue assembly; not mixed, weakened, lost or geometrically moved |
| Tissue-like background | Persistent scalar field B(q)>=0 and nonnegative R×C weights h, default zero | Same geometric map as centers; h sets destination intensities, not M or molecule loss; no new texture draw per round |
| Dependent noise | alpha>=0, default 0, intensity units | `sqrt(alpha*J)*Z_dep` for pre-noise nonnegative total J (spots+tissue+baseline); spatial/channel/round independent standard normals |
| Independent noise | sigma>=0, default 0, intensity units | `sigma*Z_ind` added after dependent term; distinct standard-normal stream, no dependence on J |

All effects have explicit enable flags default false except the scene/kernel.
Disabled means algebraic identity (gain=1, mixing=I, geometry=identity) or zero
addition regardless of retained requested parameters; record requested and
effective configurations. Clean images have no random background/noise. Changing
noise strength preserves standardized noise draws, identities, positions, shapes,
round masks and transforms. A change in upstream J appropriately changes dependent
noise amplitude, not its standardized draw. Persistent loss overrides temporary
recovery; overlapping weak/dropout flags are retained for attribution. Mixed
signal is not a new identity or a changed codeword. No carryover between rounds.

### Structured background definition

B is a sum of independently enabled reference-frame components (all default zero):

* Gradient: `max(0, a0 + dot(a, q/(shape−1)))`, with singleton normalized coordinate
  zero; a0>=0, slopes signed, intensity units. Coefficients supplied, no random draw.
* Broad regions: supplied centers, positive sigma_zyx and nonnegative peak heights;
  sum untruncated axis-aligned Gaussian bumps. Default enabled width (2,8,8),
  height 10; centers/count must be supplied. These are not segmented objects.
* Texture: count or Poisson density of mathematical blobs using the same
  count/placement conventions in an independent background namespace; default
  enabled count 4, uniform placement, width (1,3,3), height 5. Width/height
  variation uses the distributions above. Sum untruncated Gaussian blobs.

Record component IDs, centers, widths, heights and stream descriptors. Scalar B
is defined analytically on all real coordinates, avoiding a periodic wrap or
an invented extrapolated image. Evaluate B at inverse-mapped output coordinates.
Its infinite Gaussian tails and extrapolated gradient are deliberate model
semantics; amplicon kernels have the explicitly finite support above. This is a
controlled smooth background, not a claim of biological tissue texture. Baseline
is separate so geometric motion cannot move an instrument-like constant offset.

### Geometry and boundaries

Default translation t=(0,0,0), voxels; configured translation is supplied per
round or drawn componentwise U[−a,a), a>=0 (a=0 means exactly zero). Each round
uses its own label-keyed stream. Local deformation is a finite sum of Gaussian
radial basis displacements in reference coordinates:

`d(q) = sum_k v_k * exp(−||q−c_k||^2/(2*l_k^2))`; `F_r(q)=q+d_r(q)+t_r`.

Centers c are supplied; l>0 is spatial scale in voxels; v is a supplied signed
ZYX vector or independent normal components with SD `strength` (default 0).
Configured enabled default l=8; strength scales the vector, not the coordinate.
For invertibility require `sum_k ||v_k||/(l_k*sqrt(e)) <= 0.5`; reject violations
without rescaling/redrawing. This conservative Lipschitz bound makes the inverse
fixed point `q_next=p−t−d(q)` contractive. Stop at max-norm update <=1e−10 voxels,
maximum 100 iterations; failure is an explicit error. Record residual and iteration
count. This is a bounded smooth deformation family, not arbitrary elastic tissue.
Do not add a second interpolation of molecule centers. Puncta translate with
their centers while retaining local effective widths/angle: v1 does not deform
their shapes via a Jacobian. The scalar background is advected through F inverse.

Z=1 requires t_z=0, v_z=0 and c_z=0; distances then use YX with dz=0. Nonzero
out-of-plane motion errors rather than being silently dropped. No explicit
rotation, scale or shear is added. Field arrays, when saved, are ZYX3 samples of
this **forward** reference-to-round map. They are not registration pull fields.
An exact translation's registration correction is −t. For a nonlinear map,
negation is not its inverse. Any derived pull field must state source/destination
frames and be checked through composition. Identity/integer maps use exact
arithmetic where possible; generated geometry never uses periodic wrapping.

## Truth and provenance payload

Use a synthetic source record and namespaced extension to artifact v1; do not
invent a processing stage or put synthetic labels into candidate tables.
`extensions["starfinder.synthetic"]` declares contract ID/revision, truth
components, full effective configuration/order and stream descriptors. Prepared
images reference that source. Synthetic truth uses
`truth_namespace=[dataset_version,sample_id,FOV_id,"formed"]` encoded as a JSON
string. Processing candidates use their own detection namespace. Matching is an
explicit evaluation operation, never a join by coincident integer IDs.

| Record | Required fields and types |
| --- | --- |
| Formed table (N rows) | namespace/amplicon_id/gene_id/codeword/frame_id: strings; formed_index int64 0..N−1; reference z/y/x, A/sz/sl/e/theta float64; geometry reference; stable IDs independent of visibility |
| Round table (N×R rows) | namespace/ID/round_label, round_index int64; transformed z/y/x float64; transform_id; dropped/weakened/lost/emitting/center_in_bounds/support_intersects/support_truncated Boolean; first_loss_round nullable int64; trend/weak multiplier float64 |
| Signal truth | float64 N×C×R intended, pre_mix and realized amplitudes; explicit ordered labels and channel mapping; not extracted neighborhood sums |
| Geometry/background | transform coefficients/units/direction/frame IDs and any sampled fields; background component truth and weights, separate from amplicon truth |
| Observation | rendered shape/dtype, clipping counts; noise parameters/standardized stream identities; hashes and optional noise arrays; source/config/codebook/parent identities |

`center_in_bounds` uses closed voxel-center bounds 0..L−1. `support_intersects`
means at least one integer output voxel lies in the 4-sigma ellipsoid;
`support_truncated` means its continuous axis-aligned extent leaves those bounds.
Rotated lateral extents are 4*sl*sqrt(e²*cos²(theta)+sin²(theta)) in Y and
4*sl*sqrt(e²*sin²(theta)+cos²(theta)) in X. Z extent is 4*sz.
`emitting` means any realized amplitude >0; visibility is the pair
(emitting, support_intersects), not detectability or an intensity threshold.
For Z=1, truncation along Z is recorded as true, with `singleton_z_sampling=true`
in scene metadata so it is distinguishable from an accidental crop.

Keep lost, all-zero, invisible, overlapping and out-of-frame objects. N=0 retains
typed empty tables and arrays (0,C,R), not null truth. No eligibility flag is
inferred from detection or decoding. A future eligibility table needs a named,
versioned policy and reason linked to complete truth. Do not equate a known
formed object with original biological RNA or a recoverable isolated spot.

## Independent deterministic streams

The reference descriptor is a JSON array in exactly this order:

`["starfinder.synthetic/1", split, root_seed, scene_key, component, entity, round_label, channel_label]`.

All strings are nonempty NFC-normalized Unicode; optional fields are null.
`scene_key` is a stable user-supplied realization identity, not a config hash or
run timestamp. Split is development, calibration or evaluation (latter two
reserved). Seed is integer 0..2**64−1. Component is a stable registry name, not
an ordinal assigned by loop order. Serialize with UTF-8, ensure_ascii=false,
compact separators and allow_nan=false; derive SHA-256 digest. Interpret the
entire digest as an unsigned **big-endian** integer, seed NumPy PCG64 with it,
then use `numpy.random.Generator`. Save descriptor, digest, bit generator and
NumPy version. No Python hash(), global RNG, wall-clock seed or order-dependent
SeedSequence.spawn. This specifies repeatability in the pinned environment;
NumPy distribution algorithms across releases are not promised bitwise stable.

Registry: `count`, `placement`, `identity`, `brightness`, `width.axial`,
`width.lateral`, `elongation`, `angle`, `round.dropout`, `round.weakening`,
`round.loss`, `geometry.translation`, `geometry.local`, `background.count`,
`background.placement`, `background.width`, `background.brightness`,
`noise.dependent`, `noise.independent`.
Property draws are keyed by stable amplicon ID (or blob ID); round-specific draws
also use the round label. Image noise uses round+channel and C-order ZYX draw
order. Each component starts its own generator; rejection sampling consumes only
that entity's stream. Count uses null entity/round/channel. Persistent properties
use null round/channel; loss is persistent. Supplied values consume no draws.
For multi-property background widths and local control vectors, use a compact
JSON string entity `[object_id,property_name]` (e.g. `["blob-0","axial"]` or
`["control-0","vector"]`); vector draws consume Z,Y,X in that order. Blob height
uses its own brightness component. Cluster choice and its rejected proposals
consume the same per-amplicon placement stream; there is no global rejection RNG.
Changing a parameter never changes the descriptor of unrelated components;
changing shape can change noise array layout and is not a voxel-overlap guarantee.
Appending IDs cannot redraw existing entities; changing scene_key intentionally
resamples the scene. Reordering scheduling must not change draws for a given key.

## Frozen development cases and downstream acceptance

`synthetic-contract-v1` is a hand-built oracle, not a generated scientific fixture.
Clean presets `formed-small-v1` and `formed-z1-v1`: shape (8,32,32) or (1,32,32),
N=8 uniform, seed 42, scene_key `formed-v1`, split development, dataset version
matching preset name, sample `sample`, FOV `FOV_001`; float32, unknown calibration;
round labels (round10,round2,round1), channels (ch02,ch00,ch03,ch01);
color mapping 1→1,2→0,3→3,4→2; genes gene-A=123, gene-B=214 with equal abundance.
A=100, sz=sl=e=1, theta=0; all effects disabled and transforms identity. Z1 uses
the same model, not a projection of the 3D preset. These sizes do not replace
historical tiny/small presets. Only downstream W-157 generates these images.

| Case | Independent expectation (rtol=0) | Required consumer |
| --- | --- | --- |
| A1 scene | Count=0 empty; explicit coincident IDs retained; two equal weights and density=0 endpoint; invalid simultaneous count/density rejected | W-157 |
| A2 kernel | A=8, center (1,2,2), sigma=(1,1,1): center=8, one-axis distance 1 gives 4.852245277701067; at distance 4 gives 0.002683701023220095; distance>4 gives 0. Fractional center x=2.5 gives equal adjacent samples 7.059975220676764. Repeat at z=0 for Z=1 | W-157 |
| A3 shape/boundary | A=8, sz=2, sl=1, e=2, theta=0: dz=2 or dy=2 gives 4.852245277701067; dx=2 gives 1.0826822658929016. Out-of-frame center x=−0.5 still contributes 7.059975220676764 at x=0; center-in-bounds false, support-intersects true | W-157 |
| A4 histories | A=8, three rounds, b=0.5: [8,4,2]; dropout only middle gives [8,0,2]; loss starting middle gives [8,0,0]; intended remains [8,8,8]. Weak factor 0.25 middle gives [8,1,2] | W-161 (future) |
| A5 mixing | Source [8,0,0,0], M column 0=[1,0.25,0,0], remaining identity columns: result [8,2,0,0]. Source gain 0.5 gives [4,1,0,0]; add baseline [1,2,3,4] → [5,3,3,4] | W-161/W-162 (future) |
| A6 noise | J=9, alpha=4, Z_dep=−0.5, sigma=2, Z_ind=0.25 → 6.5; disabled both → 9; J=0 dependent term=0; nearest-even [−1,0.5,1.5,255.5] to uint8 → [0,0,2,255] | W-162 (future) |
| A7 geometry | q=(1,2,3), t=(0.5,−1,2) → (1.5,1,5); correction=−t. Constant d=(0,0,0.25) then t=(0,0,0.5) gives x+0.75. Nonconstant d_x=0.25*exp(−||q−c||²/8), c=(1,2,3), maps c_x to 3.25 before translation; inverse composition residual <=2e−10 | W-163 (future) |
| A8 streams | Different PYTHONHASHSEED processes produce identical descriptor digests and draws; toggle/noise amplitude/component scheduling leaves other streams exact; different split/component/entity/round/channel identities get distinct descriptors/digests | W-155 reference; W-157 actual scene; W-167 full effects |
| A9 persistence/E1 | Reload saved 3D and Z=1 images/truth/config/mappings exactly, including dtype, signed zero, nulls, IDs and lost rows; independently decode/extract reloaded checkpoint identically to uninterrupted processing | W-157/W-160, via W-158/W-159 |

For A2/A3 elementary exp calculations in float64 use atol=1e−12; float32 final
samples use atol=1e−6 (rounding of these values under 8, not arbitrary amplitudes).
Representable A4–A6 arithmetic is exact. A7 bound is inverse-solver numerical
error, not registration accuracy. Persisted values/dtypes/IDs/configs require
exact equality, never these computational tolerances. Later implementations
must assert these literals **without using their renderer to create expectations**.
Add rejection cases for malformed codebooks, nonfinite/out-of-domain parameters,
invalid geometry and missing truth keys. Full effect combinations, calibrated
ranges and scientific thresholds are unexecuted here.

W-157 intake is A1–A3, A8 and truth/provenance parts of A9. W-160 intake is A9
plus the saved 3D/Z=1 inspection packet. Future W-161–W-163 and W-166/W-167 must
cite this contract's reviewed revision and relevant cases; this matrix does not
authorize their execution or pass W-173/W-174 human review.

### Executable independent arithmetic and stream example

From `src/python` in the prepared environment:

```bash
uv run python ../../docs/examples/synthetic_specification.py
uv run pytest test/test_synthetic_specification_examples.py -v
```

The example checks literal arithmetic and a reference seed derivation, allocates
no image volume and writes no fixtures. It is deliberately outside the production
synthetic package: passing it does not certify a future renderer, effects engine,
storage round trip or empirical realism. Production code must not import this
oracle. The tests also change PYTHONHASHSEED in separate processes.

```{literalinclude} examples/synthetic_specification.py
:language: python
```
