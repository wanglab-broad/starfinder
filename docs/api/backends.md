# Dependencies and backends

The {doc}`installation guide <../get-started/installation>` lists the packaging
extras and installation commands. The tables below show which interfaces use them.

## Registration

| Interface | Dependency |
| --- | --- |
| Translation estimation | NumPy/SciPy or scikit-image |
| TPS and CPD estimation | NumPy/SciPy/scikit-image |
| Demons estimation | SimpleITK, extra `local-registration` |
| Rigid and affine estimation | itk-elastix and itk, extra `registration-elastix` |
| B-spline estimation and grid evaluation | itk-elastix, itk and SimpleITK |
| SciPy transform application | NumPy/SciPy |
| SimpleITK transform application | SimpleITK |

The {doc}`registration reference <../registration-algorithms>` links the method
configs. For installation failures, see {doc}`../troubleshooting`; for backend
execution and error details, see the
[registration specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/registration-algorithms.md).

## Other interfaces

| Interface | Dependency / extra |
| --- | --- |
| TIFF reads and writes | tifffile, in the base package |
| Checkpoint Parquet tables | pyarrow, extra `checkpoint` |
| AnnData output | anndata, extra `anndata` |
| SpatialData export | SpatialData packages, extra `spatialdata`; see the {doc}`sample-output guide <../sample-output-contract>` |
| Learned spot detection | See {doc}`../learned-detectors` |
| Segmentation methods | See the {doc}`segmentation reference <../segmentation-algorithms>` |
| Benchmark inspection figures | Matplotlib, in the base package |
| Documentation build | Base dependencies and the `docs` group |

The extras `ome` and `visualization` install bioio-ome-tiff and napari. The public
TIFF functions use tifffile; see the {doc}`I/O API <io>`.

MATLAB is a separate backend with its own {doc}`reference <matlab>`. Matching names
between the Python and MATLAB APIs do not establish numerical equivalence.
