"""Python backend: deep-tissue subtile creation.

Like gr_single_fov_subtile.py but uses deep-tissue preprocessing config
(typically no enhance_contrast, different hist_equalize settings).
Supports streaming mode.
"""

import sys
from pathlib import Path

sys.path.insert(0, snakemake.config["starfinder_path"] + "/src/python")

from starfinder.dataset import STARMapDataset

# --- Build dataset from Snakemake config ---
sdata = STARMapDataset.from_config(snakemake.config)
fov_id = snakemake.wildcards.fovID
params = snakemake.config["rules"]["deep_create_subtile"]["parameters"]

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
        ref_img=ref_img,
        mov_img=mov_img,
        hist_equalize=params.get("hist_equalize", {}).get("run", False),
        hist_equalize_ref_channel=params.get("hist_equalize", {}).get(
            "reference_channel", 0
        ),
    )
else:
    # Batch mode
    fov.load_raw_images()

    if rotate_angle:
        fov.rotate(angle=rotate_angle)

    # Deep mode: typically no enhance_contrast

    if params.get("hist_equalize", {}).get("run"):
        ref_ch = params.get("hist_equalize", {}).get("reference_channel", 0)
        fov.hist_equalize(ref_channel=ref_ch)

    if gr_params.get("run"):
        fov.global_registration(ref_img=ref_img, mov_img=mov_img)

# --- Save outputs ---
fov.save_ref_merged()
fov.create_subtiles()
