# A segmentation record of format version 1

`segmentation.json` is the record that `FOV.segment(plan, checkpoints=...)` wrote for the run `cell` at
Starfinder `12a8f14ca664dcb6d5ce93666c0d0ce2ac7159b2`, the base revision of W-891, before the input
recipe (`format_version` 1: `input.projection`, `channels[*].prepare` and `prepare_channel`). Row RE2 of
`design/validation/segmentation-input-recipe.md` reads it with `FOV.load_segmentation`
(`test_segmentation_recipe.py`, `test_re2_a_version_1_record_still_loads`).

The FOV is the development preset `clean`, size `small`, with the spacing (0.35, 0.1, 0.1) µm, after
`FOV.run` (load and translation registration) and the morphology round `morph` of
`test_segmentation_segment.py` (`add_morphology_round`). The plan imports the hand-built nuclei of
`write_mask` as `nucleus` and grows `cell` by `seeded_watershed` (sigma 0.1 µm) on
`InputChannel("composite", "morph", "ch00", prepare=CompositeConfig())`. The label and input files are
not kept (no committed TIFFs): the test writes them again from the same arrays with the package's
writers. It was written on host GP099-29C (Intel Xeon Gold 6234) in the locked environment
`dev-s213plus-20261110T0202Z` (Python 3.12.12, NumPy 2.2.6, scikit-image 0.26.0, tifffile 2026.1.28),
one CPU and one numerical thread, by the scratch script of W-891's run folder
(`runs/W-891/20261120T140418Z-71a08de1/`, `base_constants.py`, SHA-256
`26034cf5cb5130bedd73f57a3553b92241b4c05c5d4f7afb226cdd5d6e9f99f0`), which also printed the base
revision's array hashes that `test_segmentation_recipe.py` holds as constants (RC1 and PF2).
