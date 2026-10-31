"""
    The frozen MATLAB workflow (backend: matlab; D7, design/specs/orchestration-engines.md, "The MATLAB legacy
    separation"), included by the dispatcher (workflow/Snakefile). Its schema validation, MATLAB rule files, rule order
    and local rules are those of workflow/Snakefile before the dispatcher, moved here unchanged; the schema refuses the
    Python-only blocks for this backend, so none of the Python rule files is included. The target rule all stays in
    workflow/Snakefile, with this file's get_overall_output as its input (Snakemake takes the default target from the
    main file only).

    Workflow modes:
    - 'free': Mix and match rules using individual run: True/False flags
    - 'direct': Use rsf_single_fov for direct spot finding
    - 'subtile': Use subtile-based workflow (gr_single_fov_subtile → lrsf_single_fov_subtile → stitch_subtile)
    - 'deep': Use workflow optimized for deep-tissue (deep_create_subtile → deep_rsf_subtile → stitch_subtile)
"""

from snakemake.utils import validate

### ==================== [ Config Validation ] =========================

# Validate configuration against schema (catches errors early at pipeline startup)
validate(config, schema="../schemas/config.schema.yaml")

### ==================== [ Include Rule Files ] =========================

include: "common.smk"

include: "registration.smk"
include: "spot-finding.smk"
include: "segmentation.smk"
include: "stitching.smk"
include: "reads-assignment.smk"
include: "utils.smk"

### ==================== [ Dynamic Rule Order ] =========================

# Build ruleorder based on workflow_mode to avoid ambiguous rule selection
# Rules that produce the same output (e.g., _goodSpots.csv) need ordering. The same choice names the
# producer of each FOV's molecule table that the consumers read (MOLECULE_TABLE, common.smk; _legacy_molecules).

if WORKFLOW_MODE == 'subtile' or (WORKFLOW_MODE == 'free' and is_rule_enabled('gr_single_fov_subtile')):
    # Subtile workflow: subtile rules (gr/lrsf) take priority over deep rules
    ruleorder: stitch_subtile > gr_single_fov_subtile > lrsf_single_fov_subtile > deep_create_subtile > deep_rsf_subtile > rsf_single_fov
elif WORKFLOW_MODE == 'deep' or (WORKFLOW_MODE == 'free' and is_rule_enabled('deep_create_subtile')):
    # Deep workflow: deep rules take priority over subtile rules
    ruleorder: stitch_subtile > deep_create_subtile > deep_rsf_subtile > gr_single_fov_subtile > lrsf_single_fov_subtile > rsf_single_fov
elif WORKFLOW_MODE == 'direct' or (WORKFLOW_MODE == 'free' and is_rule_enabled('rsf_single_fov')):
    # Direct workflow: rsf_single_fov takes priority
    ruleorder: rsf_single_fov > stitch_subtile > gr_single_fov_subtile > lrsf_single_fov_subtile > deep_create_subtile > deep_rsf_subtile
else:
    # Default: rsf_single_fov first (simpler workflow)
    ruleorder: rsf_single_fov > stitch_subtile > gr_single_fov_subtile > lrsf_single_fov_subtile > deep_create_subtile > deep_rsf_subtile

### ==================== [ Local Rules ] =========================

# rsf_preparation writes the JSON copy of the configuration that only the MATLAB scripts read.
localrules: create_sample_maf, rsf_preparation
