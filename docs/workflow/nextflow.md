---
orphan: true
---

# Nextflow

This page moved: the {doc}`workflow guide <index>` has the Nextflow Quick start.

(workflow-nextflow-images)=
## Running the tasks in an image

Nextflow runs a task in an image only when a container engine is enabled and the task has an image. The workflow sets the
image itself, so two things give it, and nothing else does:

* `--container <image>` on the command line, the image of the tasks and of the plan and verification processes; and
* `execution.container` in the workflow configuration, which replaces `--container` for the tasks
  ({ref}`reference <workflow-keys-execution>`). The plan and verification processes take `--container` only, so give it
  as well when they should run in the image too.

A `process.container` in a `-c` file has no effect on the workflow's processes: the workflow's own `container` setting for
its `cpu` and `gpu` tasks outranks it. Before this check, an engine enabled in a `-c` file with only a `process.container`
there left every task on the host, where it failed with `starfinder: command not found`, or, on a host with `starfinder`
installed, ran that host's code without a sign.

With `docker.enabled`, `apptainer.enabled` or `singularity.enabled` true and neither `--container` nor
`execution.container` set, the run stops before the plan:

```text
The docker engine is enabled but no container image is set: give --container <image> or set execution.container in <config>; ...
```

With no engine enabled, nothing changes: the tasks run on the host, `--container` is not needed, and each task record's
`execution_context.image` is null. A task that ran in an image records it there.

| Profile | How to run the tasks in an image |
| --- | --- |
| `local` | Enable an engine in a `-c` file (`docker.enabled = true`) and give `--container <image>`, or set `execution.container`. With no engine the tasks run on the host |
| `sge` | `--container <SIF>` is required; the profile enables Apptainer. `execution.container` replaces the SIF for the tasks |
| `google-batch` | Google Batch runs every process in an image, so the check does not apply: the profile's `container` placeholders are the images, and `--container` replaces the CPU placeholder |
