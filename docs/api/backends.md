# Dependencies and backend behavior

The reference imports the real installed checkout. Building it requires the
base Python dependencies and the `docs` group, but no microscopy data, MATLAB
license, or optional algorithm packages. See [build instructions](../contributing.md).

| Interface | Dependency / behavior |
| --- | --- |
| Plain TIFF I/O | `tifffile`, installed with the base package |
| Metadata-aware OME/ImageJ reads | `tifffile` series axes with explicit ambiguous T/C/series selection |
| Global registration | NumPy/SciPy; TranslationConfig selects scipy_fft or skimage |
| TPS/CPD registration and point-set warping | NumPy/SciPy/scikit-image; no SimpleITK or external CPD package |
| Demons estimation and SimpleITK application | Lazy SimpleITK import; missing package raises `RegistrationBackendUnavailableError` when called. Z=1 inputs are estimated as 2D (pyramid in Y and X only) |
| Rigid, affine and B-spline estimation | elastix through `itk-elastix` 0.25.4 and `itk` 5.4.7 (extra `registration-elastix`), imported only when one of these methods runs; B-spline also needs SimpleITK to evaluate its grid. A missing extra raises `RegistrationBackendUnavailableError` naming `registration-elastix` |
| `FOV.register` | Specific errors propagate; no automatic fallback |
| `benchmark.run_benchmark` | Applies each result using its application_config (SciPy for TPS/CPD, SimpleITK for demons) |
| Benchmark inspection images | Base Matplotlib; file-oriented plotting selects the Agg backend |

From `src/python`, enable demons with:

```bash
uv sync --extra local-registration
```

and rigid, affine and B-spline with:

```bash
uv sync --extra registration-elastix --extra local-registration
```

The first elastix call of a process loads the ITK modules, which takes about
10 to 15 s and adds roughly 0.5 to 0.8 GB of resident memory (W-244). elastix
follows the ITK global thread count (`itk.MultiThreaderBase` or
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`); repeated calls at one thread are
bit-identical, while one-thread and multi-thread results differ. Each call logs
to a temporary directory that is removed afterwards; the per-level iterations,
final metric values and stop conditions are read from that log into
`RegistrationDiagnostics`, together with the backend versions, the effective
elastix parameter map and the spacing source. elastix never resamples the
images (`WriteResultImage=false`); the transform is applied by Starfinder.

Direct and FOV calls have no automatic fallback. Invalid configurations raise
`InvalidRegistrationConfigError` before backend execution; insufficient landmarks
raise `InsufficientLandmarksError`, a `RegistrationEstimationError`. Missing
SimpleITK or `itk-elastix` raises `RegistrationBackendUnavailableError`. A constant
reference or moving signal raises `RegistrationEstimationError("constant registration
signal")` before elastix runs, and any elastix or ITK error or non-finite parameter
becomes a `RegistrationEstimationError` with the backend message. Unknown
diagnostics are `None`; no convergence or iteration count is fabricated (the elastix
methods run a fixed number of iterations per level, so `converged` is `None`).

`DemonsConfig()` preserves the former direct demons defaults. TPS uses noise
sigma 3 and grid spacing 32; direct CPD uses 5 and 16. The FOV adapter explicitly
retains its prior CPD values 3 and 32. These are effective settings, not claims
of MATLAB numerical equivalence. No MATLAB API changes are included.

The packaging extras `ome`, `spatialdata`, and `visualization` install
`bioio-ome-tiff`, SpatialData packages, and napari respectively. Current public
Python functions do not switch to those packages automatically. In particular,
`save_volume` writes TIFF (OME-TIFF for ZYXC) through tifffile, and FOV output methods write TIFF,
CSV, text, and NPZ; there is no public SpatialData writer in this checkout.

Benchmark cases require explicit input/output roots, ownership and configuration;
there are no public institutional default paths. See [benchmark lifecycle](../benchmark.md).
Use an external supervisor for CPU/time/memory limits. A documentation build does
not run benchmarks or certify performance claims.
