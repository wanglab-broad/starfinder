"""Reproduce one §2.5 recipe comparison on a tiny synthetic preset.

Scalar background subtraction is compared on its targeted `baseline`
development condition in the two ways the evaluation uses: isolated (no
preprocessing -> scalar background alone) and as an ablation of recipe 2
(percentile normalization alone -> scalar background then percentile
normalization). It reuses the W-233 evaluation harness
(benchmarks/preprocessing_synthetic.py), so scenes, recipes, detection,
decoding and matching are exactly those of the saved evaluation, at a tiny
size: 10x32x32 voxels, 12 amplicons, uint8, one development seed and one
held-out seed. The numbers illustrate the method;
they are not evaluation results, and the 2-point threshold is provisional.

From src/python: uv run python ../../docs/examples/preprocessing_comparison.py
"""
import importlib.util
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SHAPE, COUNT, DTYPE, SEEDS = (10, 32, 32), 12, "uint8", (0, 100)
ARMS = ("none", "scalar", "pct", "r2_scalar")
COMPARISONS = (("isolated", "none", "scalar"), ("ablation", "pct", "r2_scalar"))


def harness():
    spec = importlib.util.spec_from_file_location("preprocessing_synthetic",
                                                  ROOT / "benchmarks" / "preprocessing_synthetic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    evaluation = harness()
    results = {}
    with tempfile.TemporaryDirectory(prefix="preprocessing-comparison-") as workdir:
        for seed in SEEDS:
            book, config = evaluation.scene_config("baseline", dtype=DTYPE, seed=seed, shape=SHAPE, count=COUNT)
            scene, _background, _signal, truth = evaluation.generate(book, config)
            recipes = evaluation.arms(evaluation.background_radius(config))
            for arm in ARMS:
                fov = evaluation.make_fov(scene, book, config.FOV_id, workdir)
                fov = evaluation.preprocess(fov, recipes[arm], register=False)
                rows = evaluation.sweep(fov, truth, verify=False)
                at5 = next(r for r in rows if r["threshold"] == evaluation.DEFAULT_THRESHOLD)
                results[seed, arm] = dict(max_f1=max(r["f1"] or 0.0 for r in rows),
                                          auprc=evaluation._auprc(rows), f1_t5=at5["f1"],
                                          correct_fraction_t5=at5["correct_fraction"])
    for (seed, arm), r in results.items():
        print(f"seed {seed:3d} {arm:9s} max-F1 {r['max_f1']:.3f}  AUPRC {r['auprc']:.3f}  "
              f"F1 at 5 {r['f1_t5']:.3f}  correct decodes at 5 {r['correct_fraction_t5']:.3f}")
    held_out = SEEDS[-1]
    for mode, before, after in COMPARISONS:
        b, a = results[held_out, before], results[held_out, after]
        print(f"held-out seed {held_out}, {mode} {before} -> {after}: "
              f"max-F1 delta {a['max_f1'] - b['max_f1']:+.3f}, "
              f"correct-decode delta at 5 {a['correct_fraction_t5'] - b['correct_fraction_t5']:+.3f} "
              "(provisional low-benefit threshold 0.02)")
    return results


if __name__ == "__main__":
    main()
