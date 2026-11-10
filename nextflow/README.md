# Nextflow workflow

Run the Python task manifest through the existing `starfinder workflow` commands with
Nextflow 26.10.4 or newer and Java 17 or newer. `--config` names the Python workflow
YAML, rather than a Nextflow configuration file. The `local` profile executes jobs
serially and honors their resource requests. `-stub-run` writes declared-output
placeholders under the configured output root and runs planning and both completion checks; it does no image processing.

`PLAN` always runs. Before fan-out, unsupported kinds, Fiji tasks, a fourth seed depth
and a fourth supplied-statistics step are refused. Supply `stitching.dataset_xml`
and `stitching.view_map` for a BigStitcher placement. Each producer emits one record;
`combine(by: 0)` pairs every dependency edge, including fan-out, and `groupKey`
releases a task when all of its needs have completed. Repeated statistics stages and
seed depths each have three static aliases. GPU tasks use separate static aliases.

`-resume` first runs `VERIFY_BEFORE`. A complete task keeps its recorded token; every
other task gets a fresh invocation token. Each task hashes its id, fingerprint,
token, manifest path and sorted dependency records, and passes `--cache-token` to
its command. The task entry is looked up for directives and command rendering,
outside the hashed inputs: changing memory or runtime alone keeps the cache.
The verification jobs run locally; `VERIFY_AFTER` always checks the actual output tree, including terminal outputs.
The command writes data in place and the process copies its completion record into
its work directory. Moving the manifest path still changes the cache inputs. Container identity and
CPU/GPU process aliases are intrinsic Nextflow cache keys, so changing either
reruns affected tasks.

For `sge`, supply `--container` with the shared-drive CPU SIF for planning,
verification and default task execution, and put `-work-dir` under `/stanley`.
A task's `execution.container` can override that image. The profile uses Apptainer,
Broad's per-slot `h_vmem` request, and the `smp` parallel environment for a task of
more than one CPU. `--weights_dir <folder under /stanley>`, when given, passes
`--env STARFINDER_WEIGHTS_DIR=<folder>` to every task. The GPU aliases
request RedHat8 GPU nodes and add `--nv`. Exit statuses 2 and 3 finish without retry;
other failures retry up to `--max_retries` (default 2), scaling task memory and time
by the attempt; `--max_retries 0` retries nothing and finishes the run after the failure. GPU execution and UGER submission require their separate checks.

The dedicated CI job uses Java 17 and Nextflow 26.10.4 and selects `-m nextflow`.
The tests skip with `Nextflow not installed` when the executable is absent. Set
`STARFINDER_NEXTFLOW_EVIDENCE` to retain invocation logs, traces and input inventories
outside pytest's temporary directory. Nextflow cache/work files always stay local
for these tests; output artifact copies can go to the durable run directory.
