# Broad shared environment from the lock

This is the engine record's procedure for interactive use on RHEL 8 hosts
(notebooks or a task by hand). UGER Python jobs use the image, not this environment.
No shared environment is installed or switched by the profile repair.

1. Obtain a RHEL 8 node with `qrsh -l operating_system=RedHat8 -l h_vmem=8G`.
   Keep `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` under
   `/stanley/WangLab/envs/.uv/`; install uv 0.9.28 in that folder's `bin/`.
2. Check out the revision to deploy and record it and the lock's SHA-256.
   Set `UV_PROJECT_ENVIRONMENT=/stanley/WangLab/envs/starfinder-<short commit>`.
3. From `src/python`, use uv's Python 3.12 and run
   `uv sync --locked --no-default-groups --group workflow --extra ome --extra local-registration --extra checkpoint --extra anndata --extra multiview-stitcher --extra spatialdata`.
   Add `--extra stardist --extra cellpose --extra spotiflow --extra piscis` only
   for the learned environment. For that set, add
   `--reinstall-package opencv-python-headless`: the two locked OpenCV wheels
   share `cv2/`, so the headless wheel must be installed last (W-648).
4. Run `uv pip check`, `starfinder --help` and `snakemake --version` (9.15.0).
   Write `environment.json` beside the environment with the revision, lock
   checksum, extras and `uv pip freeze` output. Verify imports for the chosen extras.
5. Point `/stanley/WangLab/envs/starfinder-current` at it only after checks and
   owner authorization for the switch; retain the previous environment until
   the replacement is accepted.

RHEL 7 CPU nodes have glibc 2.17, too old for several locked wheels. The image
brings glibc 2.36. Availability of RHEL 8 CPU nodes remains a Broad-gate question.
The MATLAB workflow keeps its frozen `starfinder-v9` environment and separate profile.
