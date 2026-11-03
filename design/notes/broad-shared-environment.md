# Broad shared environment from the lock

This is the engine record's procedure for interactive use on RHEL 8 hosts
(notebooks or a task by hand). UGER Python jobs use the image, not this environment.
No shared environment is installed or switched by the profile repair.

% Amended 2026-11-03 for W-649 (U10, run on uger-rh8-d014): `--no-editable` and `--no-bin` added, the footprint
% and RHEL 8's missing `/usr/bin/time` stated, and the RHEL 8 question settled.

1. Obtain a RHEL 8 node with `qrsh -l operating_system=RedHat8 -l h_vmem=8G`.
   Keep `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` under
   `/stanley/WangLab/envs/.uv/`; install uv 0.9.28 in that folder's `bin/`.
   Install the Python with `uv python install --no-bin 3.12`: without
   `--no-bin`, uv also writes a `python3.12` link into `~/.local/bin` of the
   small cluster home.
2. Check out the revision to deploy and record it and the lock's SHA-256.
   Set `UV_PROJECT_ENVIRONMENT=/stanley/WangLab/envs/starfinder-<short commit>`.
   A `git archive` checkout has no `.git`: set `SETUPTOOLS_SCM_PRETEND_VERSION`
   to the revision's version, as the image build does.
3. From `src/python`, use uv's Python 3.12 and run
   `uv sync --locked --no-editable --no-default-groups --group workflow --extra ome --extra local-registration --extra checkpoint --extra anndata --extra multiview-stitcher --extra spatialdata`.
   Without `--no-editable` the environment imports the package from the
   checkout, and removing the checkout breaks it.
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
brings glibc 2.36. UGER has RHEL 8 CPU nodes (`uger-rh8-d001` and up, glibc
2.28), which `-l operating_system=RedHat8` selects. The CPU environment is about
1.5 GB, and uv's cache in `.uv/` about as much again, plus about 200 MB for the
Python. RHEL 8 nodes have no `/usr/bin/time`; time a step with the shell's
`time`.
The MATLAB workflow keeps its frozen `starfinder-v9` environment and separate profile.
