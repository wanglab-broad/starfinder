# Repository Guidelines

## Project Structure & Module Organization

- `workflow/`: Snakemake pipeline (`workflow/Snakefile`) with modular rule files in `workflow/rules/*.smk`, config validation in `workflow/schemas/config.schema.yaml`, and helper scripts in `workflow/scripts/` (Python + MATLAB).
- `code-base/src/`: core MATLAB functions used by image processing steps.
- `config/`: Conda environment definitions (primary: `config/environment.yaml`).
- `example/`: demos and analysis examples (see `example/README.md`).
- `test/`: config templates/examples for running the pipeline (start from `test/minimal_config.yaml`).
- `profile/broad-uger/`: Snakemake profile for Broad/WangLab UGER cluster runs.

## Build, Test, and Development Commands

```bash
# Create and activate the main environment
conda env create -f config/environment.yaml
conda activate starfinder

# Run the workflow (requires a fully-populated config)
snakemake -s workflow/Snakefile --configfile path/to/config.yaml --cores 8

# Validate config + DAG without running jobs
snakemake -s workflow/Snakefile --configfile path/to/config.yaml -n
```

Cluster example:

```bash
snakemake -s workflow/Snakefile --configfile path/to/config.yaml --profile profile/broad-uger
```

## Coding Style & Naming Conventions

- Indentation: 4 spaces for Python/Snakemake; 2 spaces for YAML.
- Snakemake: rule names use `snake_case`; keep rule logic in `.smk` files and heavy lifting in `workflow/scripts/`.
- MATLAB: one primary function/class per file; filename matches the function/class (e.g., `LoadImageStacks.m`).
- Formatting/linting: no repo-wide tool is enforced; keep changes minimal and consistent with surrounding code.

## Testing Guidelines

- There is no dedicated unit-test suite; validation is primarily by running/dry-running the pipeline.
- For changes to rules/config, at minimum run a dry-run (`snakemake ... -n`) and ensure schema validation passes.
- If you add/rename config keys, update `workflow/schemas/config.schema.yaml` and keep `test/minimal_config.yaml` aligned.

## Commit & Pull Request Guidelines

- Commits are short and imperative; many use an optional numeric prefix (e.g., `52. fix ...`). Match the existing style.
- PRs should include: what workflow mode/rules changed, any config/schema updates, and how to reproduce (command + config snippet).
- Do not commit generated artifacts or large data (e.g., `.snakemake/`, `workflow/.snakemake/`, `.tif`); these are gitignored.

