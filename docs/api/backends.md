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
| Demons estimation and SimpleITK application | Lazy SimpleITK import; missing package raises `RegistrationBackendUnavailableError` when called |
| `FOV.local_registration` | Specific errors propagate; no automatic fallback |
| `RegistrationBenchmarkRunner.run_local_benchmark` | Applies each result using its application_config (SciPy for TPS/CPD, SimpleITK for demons) |
| Benchmark inspection images | Base Matplotlib; file-oriented plotting selects the Agg backend |
| `timeout_handler` | Unix `SIGALRM`; no-op on platforms without it. Use in the main thread; nesting does not preserve an earlier alarm timer |

From `src/python`, enable demons with:

```bash
uv sync --extra local-registration
```

Direct and FOV calls have no automatic fallback. Invalid configurations raise
`InvalidRegistrationConfigError` before backend execution; insufficient landmarks
raise `InsufficientLandmarksError`, a `RegistrationEstimationError`. Missing
SimpleITK raises `RegistrationBackendUnavailableError`. Unknown diagnostics are
`None`; no convergence or iteration count is fabricated.

`DemonsConfig()` preserves the former direct demons defaults. TPS uses noise
sigma 3 and grid spacing 32; direct CPD uses 5 and 16. The FOV adapter explicitly
retains its prior CPD values 3 and 32. These are effective settings, not claims
of MATLAB numerical equivalence. No MATLAB API changes are included.

The packaging extras `ome`, `spatialdata`, and `visualization` install
`bioio-ome-tiff`, SpatialData packages, and napari respectively. Current public
Python functions do not switch to those packages automatically. In particular,
`save_volume` writes TIFF through tifffile, and FOV output methods write TIFF,
CSV, text, and NPZ; there is no public SpatialData writer in this checkout.

Benchmark constants include institutional default paths. Pass `data_dir`,
`results_dir`, and output paths explicitly on other hosts. Synthetic preset
names and registration benchmark size presets are separate inventories; inspect
their generated tables before scheduling a run. A documentation build does not
run benchmarks or certify their performance claims.
