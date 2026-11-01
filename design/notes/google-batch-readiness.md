# Google Batch readiness

W-651 prepares an offline profile; it authorizes no cloud use or image publication.
The accepted [engine record](../specs/orchestration-engines.md#google-batch-readiness)
sets the requirements. The site workflow guide is a separate W-739 deliverable.

## Profile and paths

`nextflow/conf/google_batch.config` defines the `google-batch` profile with explicit
project, bucket, run and Artifact Registry digest placeholders, region `us-central1`,
spot VMs and the inherited two retries. Input and refusal exits 2 and 3 still finish
without retry. `FOV` and `TILE_GLOBAL` use `n2-highmem-4`; other CPU tasks use
`n2-standard-4`; the GPU label selects `g2-standard-4`, one NVIDIA L4 and a separate
GPU image digest placeholder. W-648's learned image is CPU-only; a GPU image needs
its own lock/image decision before this label can run on a GPU. CPU learned tasks
select the learned image through their manifest container setting.
Manifest CPUs, memory and time still apply; the owner must check that
each request fits its machine, disk and regional quota before submission. Explicit
manifest container settings must also name the authorized Artifact Registry digest.

| Purpose | Convention |
| --- | --- |
| Input mirror | `gs://<bucket>/data/<dataset_id>/<sample_id>/`, preserving `root_input_path/dataset_id/sample_id` |
| Durable result mirror | `gs://<bucket>/results/<dataset_id>/<output_id>/`, preserving the output root's internal layout |
| Nextflow work | `gs://<bucket>/work/<run>/`, with a unique run identifier; remove only after acceptance |
| CPU, learned CPU and future GPU images | `us-central1-docker.pkg.dev/<project>/<repository>/starfinder@sha256:<digest>`, separately authorized digests |

Offline profile resolution needs no project, credentials or plugin download:

```bash
export NXF_OFFLINE=true NXF_PLUGINS_DEFAULT=false
nextflow config -profile google-batch nextflow/main.nf
```

`NXF_PLUGINS_DEFAULT=false` disables automatic cloud-plugin loading for these
offline checks; otherwise Nextflow tries to download its uncached `nf-google`
plugin even for `config`. This is configuration inspection only. An authorized
cloud environment must provision the matching `nf-google` plugin and allow it to
load; offline checks do not exercise that plugin.

N1 uses `-profile google-batch,local -work-dir <local-disk-path> -stub-run`.
The second profile overrides the executor; the explicit local work directory avoids
GCS, and no local container runtime is enabled. This checks graph ordering and stub
records under the cloud process settings, not Batch submission or container execution.

## Output-root decision

Choose a POSIX local or persistent disk for `roots.output`, copying completed task
products to the result mirror after each task; do not use Cloud Storage FUSE for
in-place writes. HDF5 window writes, atomic rename and the contract's completion
records require disk semantics. A GCS URI is a mirror, never the Python output root.

The tiled route requires a persistent disk: keep the chunked checkpoint and tile
products across `tile_global`, all `tile` readers, `tile_merge` and preemption.
Parallel readers must see the same checkpoint bytes; dependent writers must be
scheduled with exclusive writable attachment, or copies materialized on their own
disks. Never assume a persistent disk supports concurrent writable mounts on
independent Batch VMs. Keep the disk until the run is accepted and verified in GCS.

For a non-tiled task, local scratch is allowed only if it can restore every needed
input and completed output on retry. Stage dependencies from the mirror before the
command, copy all task products before its completion record, verify checksums, and
publish that record last. A failed transfer must fail the task; an incomplete GCS
prefix must never count as completion. Do not synchronize the entire shared output
tree from concurrent workers or propagate deletions from one worker to the mirror.

This is the storage route decision, not an implemented transfer adapter. Today's
`PLAN`, `VERIFY_BEFORE`, `VERIFY_AFTER`, manifest resource lookups and `TaskCommands`
read local absolute paths, and the verify processes explicitly run on the head host.
Before a first cloud run, provision the disk/mount route, stage inputs and manifests
at the paths recorded in the manifest, make outputs visible to the head verifier,
and implement and test the per-task restore/copy/completion ordering above. The
profile alone does none of these steps. Its real `script:` path remains untested
until an authorized cloud run, including preemption, HDF5 and transfer recovery.

## Owner checklist before the first cloud run

Each row names a decision owner; no value or approval is supplied by this readiness
change. Jiahao records the decisions and links the access grants and estimates in
Linear before requesting a cloud run. The rows cover every item of the engine
record's readiness table.

| Area | Item to record | Who decides or grants it |
| --- | --- | --- |
| Access | GCP project and linked billing account | Jiahao with the GCP project and billing administrators |
| Access | Batch service account; Batch Job Editor, Service Account User, Logs Writer, Storage Object Admin scoped to the bucket, Artifact Registry Reader | Project IAM administrator, with Jiahao confirming the submitting and runtime identities and least-privilege scopes |
| Access | Batch, Compute, Storage and Artifact Registry APIs enabled | Project administrator |
| Authorization | Publication of each CPU/learned image to Artifact Registry by digest, D9 | Jiahao explicitly authorizes publication; registry administrator grants access |
| Authorization | A GPU-capable lock/image and its publication before GPU tasks, since the existing learned image is CPU-only | Jiahao approves the separate GPU lock/image decision and publication; image owner verifies GPU support |
| Authorization | Which data may leave Broad; real data's lab data-use approval | Jiahao and the lab's data-use approver |
| Budget | Project budget, alert thresholds, recipients and stop owner | Jiahao with billing administrator |
| Budget | Regional CPU, memory and GPU quotas, including L4/G2 availability | Project quota administrator, with Jiahao choosing the region and sizing |
| Budget | Spot or standard VMs and the retry/preemption policy | Jiahao; the readiness default is spot with two retries |
| Budget | Per-FOV estimate from task `wall_s` and `peak_rss_kib`, disk, retries, storage and transfer charges | Jiahao, using the tiled resource model and current prices before authorization |
| Data transfer | `/stanley` to GCS transfer plan with `gcloud storage rsync`, total bytes and expected time; aging baseline 5.2 GiB per FOV is W-641's estimate, not a new measurement | Jiahao with Broad storage/network and lab data-use owners |
| Data transfer | Result egress back to Broad, expected bytes, time and cost | Jiahao with Broad storage/network owner |
| Data transfer | Bucket location matches the Batch region, avoiding cross-region transfer charges | Jiahao with GCP storage administrator |
| Data transfer | Lifecycle rule for `work/`, acceptance before removal, result and disk retention | Jiahao with GCP storage administrator |
| Records | Task record `image` is the executed Artifact Registry digest; `host` identifies the Batch VM | Jiahao verifies captured records on the first authorized run |
| Records | Record `wall_s` and cost per task kind, including retries, disk and transfer attribution | Jiahao reviews the task records and billing export |
| Storage prerequisite | Disk attachment, checkpoint persistence, head-host visibility, per-task restoration and verified copy before publishing completion | Jiahao with the cloud implementation owner; test before real data or a tiled cloud run |

Do not obtain or write credentials, enable APIs, create resources, transfer data or
push an image while performing the offline readiness checks.
