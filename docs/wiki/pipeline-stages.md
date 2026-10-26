# Pipeline stages

Each stage of the pipeline has a guide: its contract page, which states the stage's
entries, configurations, records, limits and expected failures. Beneath each guide are
the stage's other pages:

* **Algorithms:** the specification of each method. The validation designs of
  registration, spot finding, readout, assignment (§2.9) and the processing report are
  sections of these pages.
* **Validation:** a separate validation design, where the stage has one. The §2.10 design
  covers placement, assembly and the sample outputs, and sits under the sample-output
  guide, the last stage it checks.
* **Baseline:** how the existing code behaved before the stage's contract, pinned to the
  revision it describes. A baseline is history, not a description of the current code.

The stages are listed in processing order: the per-FOV stages, then the sample stages.

```{toctree}
:maxdepth: 2
:titlesonly:
:includehidden:

Preprocessing <../preprocessing-contract>
Registration <../registration-contract>
Spot finding <../spot-finding-contract>
Readout <../readout-contract>
Segmentation <../segmentation-contract>
Assignment <../assignment-contract>
Stitching <../stitching-contract>
Assembly <../assembly-contract>
Sample output <../sample-output-contract>
Sample segmentation <../sample-segmentation-contract>
Processing report <../report-contract>
```
