starfinder.benchmark
====================

Measurement, reporting, registration evaluation, and data helpers. Timings are seconds; ``memory_mb`` is traced allocation peak divided by 1024², not total process RSS. Paths defaulting to institutional storage must be overridden on other hosts. Synthetic generators are on :doc:`synthetic`. Pure metrics are on :doc:`evaluation.registration`.

.. currentmodule:: starfinder.benchmark

.. autosummary::
   :toctree: generated

   BenchmarkResult
   BenchmarkSuite
   benchmark
   measure
   run_comparison
   run_benchmark
   RegistrationBenchmarkRunner
   RegistrationResult
   BenchmarkPair
   timeout_handler
   generate_inspection_image
   generate_overview_grid
   extract_real_benchmark_data
   evaluate_directory
   evaluate_single
   generate_inspection
   print_table
   save_csv
   save_json

.. autodata:: starfinder.benchmark.runner.PRESET_ORDER

.. autodata:: starfinder.benchmark.presets.DEFAULT_BENCHMARK_DIR

.. autodata:: starfinder.benchmark.presets.BENCHMARK_TASK

.. autodata:: starfinder.benchmark.data.REAL_DATASETS
