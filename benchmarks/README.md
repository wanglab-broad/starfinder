# Benchmark recipes

Maintained source-derived configuration and adapters for the common
`starfinder benchmark run/evaluate/report` lifecycle. Read
[the recipe guide](../docs/benchmark-recipes.md) before selecting a profile.

- `recipes.py`: build an explicit registration or pipeline case, without data discovery.
- `configs/registration.json`: distinct method parameter profiles.
- `configs/pipeline.json`: source-specific processing/residency profiles.
- `configs/source-map.json`: complete bounded 56-source disposition and hash index.
- `report_saved.py`: saved-table reporting with explicit join keys and resource scopes.

Original scripts, results and restricted reference snapshots remain external.
These new implementations follow the Python package's declared MIT terms;
no external source license or redistribution right is inferred. All generated
configs, inputs, runs and reports belong outside this checkout. Real/large,
Postcode, hybrid and MATLAB experiments require their documented prerequisites
and separate execution scope. No default sweep is provided.
