# Dependencies and backend behavior

The reference imports the real installed checkout. Building it requires the
base Python dependencies and the `docs` group, but no microscopy data, MATLAB
license, or optional algorithm packages. See [build instructions](../contributing.md).

| Interface | Dependency / behavior |
| --- | --- |
| Plain TIFF I/O | `tifffile`, installed with the base package |
| Metadata-aware OME/ImageJ reads | `bioio` plus explicitly selected `bioio_tifffile.Reader`, both base dependencies |
| Global registration | NumPy/SciPy; comparison wrapper uses scikit-image |
| TPS/CPD registration and point-set warping | NumPy/SciPy/scikit-image; no SimpleITK or external CPD package |
| Demons registration and `apply_deformation` | Lazy SimpleITK import; missing package raises `ImportError` when called |
| `FOV.local_registration(method='tps'/'cpd', fallback=True)` | Any `ValueError` from the selected registration call triggers a demons retry; this then requires SimpleITK |
| `RegistrationBenchmarkRunner.run_local_benchmark` | Applies fields through `apply_deformation`, so even a TPS/CPD method needs SimpleITK for this runner's warp step |
| Benchmark inspection images | Base Matplotlib; file-oriented plotting selects the Agg backend |
| `timeout_handler` | Unix `SIGALRM`; no-op on platforms without it. Use in the main thread; nesting does not preserve an earlier alarm timer |

From `src/python`, enable demons with:

```bash
uv sync --extra local-registration
```

Setting `fallback=False` on FOV TPS/CPD calls propagates their `ValueError`
instead of attempting demons. Direct `tps_register`, `cpd_register` and their
multi-channel wrappers have no automatic fallback. Importing the registration
subpackage is possible without SimpleITK because only the calls import it.
`matlab_compatible_config()` returns a parameter dictionary without needing
SimpleITK; it describes parameter choices, not a claim of cross-backend numerical
equivalence. Demons uses unit image spacing. Unknown demons method names raise
`ValueError` once SimpleITK is available.

The packaging extras `ome`, `spatialdata`, and `visualization` install
`bioio-ome-tiff`, SpatialData packages, and napari respectively. Current public
Python functions do not switch to those packages automatically. In particular,
`save_stack` writes TIFF through tifffile, and FOV output methods write TIFF,
CSV, text, and NPZ; there is no public SpatialData writer in this checkout.

Benchmark constants include institutional default paths. Pass `data_dir`,
`results_dir`, and output paths explicitly on other hosts. Synthetic preset
names and registration benchmark size presets are separate inventories; inspect
their generated tables before scheduling a run. A documentation build does not
run benchmarks or certify their performance claims.
