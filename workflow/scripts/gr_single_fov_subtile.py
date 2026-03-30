"""Python backend: global registration + subtile creation.

Pipeline: load -> preprocess -> global register -> create subtiles (NPZ).
Supports streaming mode for lower peak memory on large FOVs.
"""

import sys
from pathlib import Path

sys.path.insert(0, snakemake.config["starfinder_path"] + "/src/python")

from starfinder.dataset import STARMapDataset

# --- Build dataset from Snakemake config ---
sdata = STARMapDataset.from_config(snakemake.config)
fov_id = snakemake.wildcards.fovID
params = snakemake.config["rules"]["gr_single_fov_subtile"]["parameters"]

fov = sdata.fov(fov_id)

streaming = params.get("streaming", False)
rotate_angle = snakemake.config.get("rotate_angle")

# Parse global registration image mode
gr_params = params.get("global_registration", {})
ref_img = gr_params.get("ref_img", "merged-image")
mov_img = gr_params.get("mov_img", "merged-image")
ref_img = "merged" if ref_img == "merged-image" else "single-channel"
mov_img = "merged" if mov_img == "merged-image" else "single-channel"

if streaming:
    fov.run_streaming_gr(
        rotate_angle=rotate_angle,
        snr_threshold=params.get("snr_threshold"),
        ref_img=ref_img,
        mov_img=mov_img,
        hist_equalize=params.get("hist_equalize", {}).get("run", False),
        hist_equalize_ref_channel=params.get("hist_equalize", {}).get(
            "reference_channel", 0
        ),
        morph_recon=params.get("morph_recon", {}).get("run", False),
        morph_recon_radius=params.get("morph_recon", {}).get("radius", 3),
    )
else:
    # Batch mode
    fov.load_raw_images()

    if rotate_angle:
        fov.rotate(angle=rotate_angle)

    if params.get("enhance_contrast", {}).get("run"):
        fov.enhance_contrast(snr_threshold=params.get("snr_threshold"))

    if params.get("hist_equalize", {}).get("run"):
        ref_ch = params.get("hist_equalize", {}).get("reference_channel", 0)
        fov.hist_equalize(ref_channel=ref_ch)

    if params.get("morph_recon", {}).get("run"):
        fov.morph_recon(radius=params["morph_recon"].get("radius", 3))

    if gr_params.get("run"):
        fov.global_registration(ref_img=ref_img, mov_img=mov_img)

# --- Save outputs ---
fov.save_ref_merged()
fov.create_subtiles()
fov.save_log(log_type="gr")
