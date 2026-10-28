# Learned detectors: install, weights and run

Spot finding has two learned detectors, Spotiflow (`SpotiflowConfig`) and Piscis
(`PiscisConfig`). This page shows how to install their extras, fetch their pretrained
weights and run them. The {doc}`spot-finding guide <spot-finding-contract>` describes
the stage, and the {doc}`spot-finding reference <spot-finding-algorithms>` lists each
config's options. This page does not recommend a method or a model; the model names
in the examples stand for any row of the table below.

## Install

Each detector has its own optional extra in `src/python/pyproject.toml`:

| Extra | Installs |
| --- | --- |
| `spotiflow` | spotiflow 0.6.5, torch 2.7.1 and torchvision 0.22.1 |
| `piscis` | piscis 1.1.0, torch 2.7.1 and torchvision 0.22.1 |

Run these commands from `src/python`, naming one or both extras:

```bash
cd src/python
uv sync --locked --extra spotiflow --extra piscis
```

`uv sync` installs only the extras it is given. Name every extra you need, for
example `--extra local-registration`, on the same command and on every later
`uv sync`. Run the commands below with `uv run` from `src/python`.

* **Python versions.** Use Python 3.11 to 3.13: torch 2.7.1 has no wheel for
  Python 3.14, so there the extras install nothing.
* **CPU-only torch.** On Linux, uv installs the CPU build, `torch 2.7.1+cpu`. Spot
  finding runs on the CPU.
* **Without the extras.** You can import `starfinder.spot_finding` and construct the
  configs without them; running a learned detector needs its extra.

## Weights

Starfinder knows six pretrained models, listed in `KNOWN_MODELS`
(`starfinder.spot_finding`). Each config names its model; there is no default.

| Method | Model | Network | Images | Download (bytes) | Source |
| --- | --- | --- | --- | --- | --- |
| `spotiflow` | `synth_3d` | 3D | Volumes | 263,107,693 | `spotiflow-models` release 0.6.0 |
| `spotiflow` | `smfish_3d` | 3D | Volumes | 263,106,200 | `spotiflow-models` release 0.6.0 |
| `spotiflow` | `general` | 2D | Single planes (Z=1) | 87,885,382 | `spotiflow-models` release 0.6.0 |
| `spotiflow` | `hybiss` | 2D | Single planes (Z=1) | 87,945,684 | `spotiflow-models` release 0.6.0 |
| `piscis` | `20230905` | 2D | Single planes and volumes | 30,077,822 | Hugging Face `wniu/Piscis` at `9bdefc72cb` |
| `piscis` | `20251212` | 2D | Single planes and volumes | 30,143,014 | Hugging Face `wniu/Piscis` at `9bdefc72cb` |

Choose a Spotiflow model of the image's dimensionality. `starfinder weights list`
prints the full SHA-256 values, and `KNOWN_MODELS` holds them.

### Commands

```bash
uv run starfinder weights list
uv run starfinder weights fetch spotiflow smfish_3d
uv run starfinder weights fetch piscis 20251212
uv run starfinder weights verify
uv run starfinder weights verify piscis 20251212
```

* `starfinder weights fetch <method> <model>` downloads a model, checks its hashes
  and installs it in `<root>/<method>/<model>/`, with a `starfinder-weights.json`
  record of the per-file hashes. It prints the model folder. The
  [specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/spot-finding.md#fetching-cache-and-verification)
  says what it does with a model folder that already exists.
* `starfinder weights list` prints the cache root and one line per model: method,
  model, dimensionality, size, SHA-256, revision and the local state. The state
  `fetched` means the folder holds `starfinder-weights.json`, `incomplete` means
  the folder exists without it, and `not fetched` means there is no folder. `list`
  hashes nothing, so `fetched` does not mean verified.
* `starfinder weights verify` re-hashes the files of each known model that has a
  folder in the cache, and `starfinder weights verify <method> <model>` those of one
  model. Each model prints `verified` or `failed` with the error, and the command
  exits with status 1 when one fails.
* The Python equivalents are `fetch_model(method, model, *, directory=None)` and
  `resolve_model`; {py:func}`~starfinder.spot_finding.resolve_model` says which files
  it checks.
* The segmentation models share the cache and the three commands
  ({doc}`segmentation-contract`, "Models").

### Cache location

Each command takes `--dir DIR`. Without it, the cache root is
`STARFINDER_WEIGHTS_DIR` when that is set, otherwise `$XDG_CACHE_HOME/starfinder/weights`,
by default `~/.cache/starfinder/weights`. Starfinder does not use the libraries' own
caches (`~/.spotiflow`, `~/.piscis/models` and the Hugging Face cache).

Detection has no directory option: it reads the weights from
`STARFINDER_WEIGHTS_DIR` or the default root. If you fetched with `--dir DIR`, set
`STARFINDER_WEIGHTS_DIR=DIR` before you run a detection.

### Detection and the cache

Detection does not download: `find_spots`, `FOV.find_spots`, `FOV.run` and the
workflow rules read the weights from the cache and check their SHA-256 before they
build the model. A loaded model is reused within the same process.

### Troubleshooting

| Error | When | What to do |
| --- | --- | --- |
| `SpotFindingBackendUnavailableError` | The method's extra is not installed | Install the extra (see "Install") |
| `MissingModelError` | The weights are not in the cache | Run the fetch command the message names, into the same root |
| `ModelHashMismatchError` | A weights file differs from the table | Remove the model folder and fetch again |
| `FileExistsError` | `fetch` finds a model folder without its `starfinder-weights.json` record | Remove the folder and fetch again |
| `IncompatibleGeometryError` | The model does not fit the image | Choose a model of the image's dimensionality |
| `ValueError` | The config names a model the method does not know | Use a model from the table |

The [specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/spot-finding.md#errors) gives the exact conditions.

## Run

Name the model, and leave `device` and `scale` at their defaults.

In Python, put the config in a `PipelineConfig` and pass an `ExecutionConfig` to
`FOV.run`:

```python
from starfinder.dataset import ExecutionConfig, PipelineConfig
from starfinder.spot_finding import PiscisConfig, SpotiflowConfig

execution = ExecutionConfig(device="cpu")
spotiflow = PipelineConfig(spot_finding=SpotiflowConfig(model="smfish_3d"))
piscis = PipelineConfig(spot_finding=PiscisConfig(model="20251212"))
# Add the loading, preprocessing, registration and later stages your run needs, then:
# fov.run(spotiflow, execution=execution)
```

For one image, call `find_spots` directly:

```python
from starfinder.image import ImageMetadata
from starfinder.spot_finding import SpotiflowConfig, find_spots

result = find_spots(image, config=SpotiflowConfig(model="smfish_3d"),
                    metadata=ImageMetadata("fov001"), spot_namespace="dataset/fov001", device="cpu")
```

In the workflow YAML, with `backend: python`, the `spot_finding` block of a Python
rule names the method and the model, and the rule-level `device` key sets
`ExecutionConfig.device`:

```yaml
backend: python
rules:
  rsf_single_fov:
    parameters:
      device: cpu
      spot_finding:
        run: true
        ref_round: round1
        method: spotiflow
        model: smfish_3d
```

For Piscis, use `method: piscis` and a Piscis model in quotes, for example
`model: "20251212"`; YAML reads an unquoted `20251212` as an integer. The other
fields keep their defaults when omitted. See {doc}`workflow-configuration` ("Spot-finding method key")
for `channel_overrides` and `rounds`.

Starfinder does not change thread settings. To run on one thread, set
`OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`NUMEXPR_NUM_THREADS` and `NUMBA_NUM_THREADS` likewise, before you start Python.

## Tests

The GitHub CI workflow does not run the learned-detector tests. To run them, install
the extras and fetch the six models. The test modules, their tiers and the commands
that run them are in
[`design/notes/learned-detectors-tests.md`](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/learned-detectors-tests.md).
