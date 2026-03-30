"""Python backend: deep-tissue RSF on a single subtile.

Pipeline: load subtile NPZ -> morph_recon -> spot find -> extract -> filter.
Like lrsf_single_fov_subtile.py but reads from deep_rsf_subtile config.
"""

import sys
from pathlib import Path

sys.path.insert(0, snakemake.config["starfinder_path"] + "/src/python")

from starfinder.dataset import STARMapDataset
from starfinder.dataset.fov import FOV

# --- Build dataset from Snakemake config ---
sdata = STARMapDataset.from_config(snakemake.config)
fov_id = snakemake.wildcards.fovID
n_subtile = snakemake.wildcards.n_subtile
params = snakemake.config["rules"]["deep_rsf_subtile"]["parameters"]

# Load codebook
codebook_path = Path(snakemake.input[1])  # genes.csv
split_index = params.get("load_codebook", {}).get("split_index") or None
sdata.load_codebook(codebook_path, split_index=split_index)

# Load subtile from NPZ
subtile_path = Path(snakemake.input[2])  # subtile_data_{n_subtile}.npz
fov = FOV.from_subtile(subtile_path, sdata, fov_id)

# --- Processing pipeline ---
if params.get("morph_recon", {}).get("run"):
    fov.morph_recon(radius=params["morph_recon"].get("radius", 3))

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
subtile_dir = subtile_path.parent
out_csv = subtile_dir / f"subtile_goodSpots_{n_subtile}.csv"
if fov.good_spots is not None and not fov.good_spots.empty:
    out = fov.good_spots[["x", "y", "z", "gene"]].copy()
    for col in ("x", "y", "z"):
        out[col] = out[col] + 1  # 0-based -> 1-based
    out.to_csv(out_csv, index=False)
else:
    import pandas as pd
    pd.DataFrame(columns=["x", "y", "z", "gene"]).to_csv(out_csv, index=False)

fov.save_score_log(suffix=f"_{n_subtile}")
