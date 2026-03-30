"""Python backend: single-FOV registration + spot finding (direct mode).

Invoked via Snakemake script: directive. Receives the `snakemake` object
with input, output, config, and wildcards.
"""

import sys
from pathlib import Path

sys.path.insert(0, snakemake.config["starfinder_path"] + "/src/python")

from starfinder.dataset import STARMapDataset

# --- Build dataset from Snakemake config ---
sdata = STARMapDataset.from_config(snakemake.config)
fov_id = snakemake.wildcards.fovID
params = snakemake.config["rules"]["rsf_single_fov"]["parameters"]

# Load codebook
codebook_path = Path(snakemake.input[1])  # genes.csv
split_index = params.get("load_codebook", {}).get("split_index") or None
sdata.load_codebook(codebook_path, split_index=split_index)

fov = sdata.fov(fov_id)

streaming = params.get("streaming", False)

if streaming:
    # Streaming mode: one round at a time, ~50% less memory
    local_method = None
    local_kwargs = {}
    lr_params = params.get("local_registration", {})
    if lr_params.get("run"):
        local_method = lr_params.get("method", "demons")

    fov.run_streaming(
        rotate_angle=snakemake.config.get("rotate_angle"),
        snr_threshold=params.get("snr_threshold"),
        intensity_estimation=params.get("spot_finding", {}).get(
            "intensity_estimation", "noise"
        ),
        intensity_threshold=params["spot_finding"]["intensity_threshold"],
        voxel_size=tuple(params["reads_extraction"]["voxel_size"]),
        end_bases=params["reads_filtration"].get("end_base"),
        start_base=params["reads_filtration"].get("start_base", "C"),
        local_method=local_method,
        local_kwargs=local_kwargs,
    )
else:
    # Batch mode: load all rounds, then process
    fov.load_raw_images()

    rotate_angle = snakemake.config.get("rotate_angle")
    if rotate_angle:
        fov.rotate(angle=rotate_angle)

    if params.get("enhance_contrast", {}).get("run"):
        fov.enhance_contrast(snr_threshold=params.get("snr_threshold"))

    if params.get("hist_equalize", {}).get("run"):
        ref_ch = params.get("hist_equalize", {}).get("reference_channel", 0)
        fov.hist_equalize(ref_channel=ref_ch)

    if params.get("morph_recon", {}).get("run"):
        fov.morph_recon(radius=params["morph_recon"].get("radius", 3))

    if params.get("global_registration", {}).get("run"):
        gr_params = params["global_registration"]
        ref_img = gr_params.get("ref_img", "merged-image")
        mov_img = gr_params.get("mov_img", "merged-image")
        # Normalize names: Snakemake config uses "merged-image", FOV uses "merged"
        ref_img = "merged" if ref_img == "merged-image" else "single-channel"
        mov_img = "merged" if mov_img == "merged-image" else "single-channel"
        fov.global_registration(ref_img=ref_img, mov_img=mov_img)

    if params.get("local_registration", {}).get("run"):
        method = params["local_registration"].get("method", "demons")
        fov.local_registration(method=method)

    if params.get("spot_finding", {}).get("run"):
        fov.spot_finding(
            intensity_estimation=params["spot_finding"].get(
                "intensity_estimation", "noise"
            ),
            intensity_threshold=params["spot_finding"]["intensity_threshold"],
        )

    if params.get("reads_extraction", {}).get("run"):
        fov.reads_extraction(
            voxel_size=tuple(params["reads_extraction"]["voxel_size"])
        )

    if params.get("reads_filtration", {}).get("run"):
        fov.reads_filtration(
            end_bases=params["reads_filtration"].get("end_base"),
            start_base=params["reads_filtration"].get("start_base", "C"),
        )

# --- Save outputs ---
fov.save_ref_merged()
fov.save_signal(slot="goodSpots")
fov.save_log(log_type="rsf")
fov.save_score_log()
