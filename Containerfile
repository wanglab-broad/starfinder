# The workflow image from uv.lock (design/specs/orchestration-engines.md, "The image").
#
# Three targets from the same lock: `cpu`, the default image (the CPU extras and the `workflow` group);
# `learned`, which adds the learned extras on CPU for `segment` and the learned detectors (W-646); and
# `learned-cuda`, which adds the lock's `cuda` extra instead: the same learned backends with torch and torchvision
# from the PyTorch cu126 index and TensorFlow with its CUDA libraries (W-792). The CUDA runtime comes from NVIDIA's
# wheels and the driver from the host (`apptainer exec --nv`, `docker run --gpus`).
# The build context has no .git: the version is `git describe`'s on the build host, passed in, and the full commit, the
# dirty flag and the lock's SHA-256 are passed in too; the build stops if the lock differs or the commit is not 40
# hexadecimal digits. The commit and the flag go into the labels and into `starfinder/_build_info.py`, which
# `starfinder._records.code()` reads, so that the image reports the same `version` and `git_commit` as a clean checkout
# of that commit (W-827).
#
# Run these commands from the repository root:
#
#   docker buildx build --file Containerfile --builder <docker-container builder> --target cpu \
#     --build-arg STARFINDER_VERSION=<version> --build-arg STARFINDER_REVISION=<full commit> \
#     --build-arg STARFINDER_DIRTY=<true|false> \
#     --build-arg UV_LOCK_SHA256=<sha256 of src/python/uv.lock> \
#     --output type=oci,dest=starfinder-cpu-<version>.oci.tar .
#
# The two targets that hold `piscis` build the same way, with their own target and archive name:
#
#   docker buildx build --file Containerfile --builder <docker-container builder> --target learned \
#     --build-arg STARFINDER_VERSION=<version> --build-arg STARFINDER_REVISION=<full commit> \
#     --build-arg STARFINDER_DIRTY=<true|false> \
#     --build-arg UV_LOCK_SHA256=<sha256 of src/python/uv.lock> \
#     --output type=oci,dest=starfinder-learned-<version>.oci.tar .
#
#   docker buildx build --file Containerfile --builder <docker-container builder> --target learned-cuda \
#     --build-arg STARFINDER_VERSION=<version> --build-arg STARFINDER_REVISION=<full commit> \
#     --build-arg STARFINDER_DIRTY=<true|false> \
#     --build-arg UV_LOCK_SHA256=<sha256 of src/python/uv.lock> \
#     --output type=oci,dest=starfinder-learned-cuda-<version>.oci.tar .
#
# deconwolf v0.4.5 (the deconvolution stage, design/specs/deconvolution.md, decision 6) is built from source in the
# stage `build-deconwolf`, from the tag's archive pinned by its SHA-256, and copied with its libraries into
# `runtime`, so that `cpu`, `learned` and `learned-cuda` all have `dw`. The build is reproducible (W-936): the stage sets
# SOURCE_DATE_EPOCH to the v0.4.5 commit's time, so that GCC's `__DATE__` (and so the binary and its build-id) no longer
# follows the build's day, and takes its apt packages from one snapshot.debian.org timestamp. Builds on different UTC days
# then give one `dw`, whose SHA-256 is the default of `DECONWOLF_SHA256` below: the wisdom store's key holds it
# (design/specs/deconvolution.md, "FFTW wisdom"), so a binary that changes with every build would make every stored plan
# unmatched. The `runtime` stage checks the binary against that argument, so a build whose `dw` differs stops there. The
# coordinating session's probe (runs/W-936/20261124T0315Z-repro-probe) showed it, and IM5 of
# design/validation/deconvolution.md repeats it: two `--no-cache` builds of the stage on different UTC days give `dw` equal
# to the default (95bb636d…) and `libtrafo.so` aea01a98….
#
# When the recipe changes on purpose (a new deconwolf version, a new snapshot date, new compiler flags), rebuild the stage
# and take the new SHA-256, then update the default of `DECONWOLF_SHA256`, and plan the wisdom store again, because its
# keys name the old binary:
#
#   docker buildx build --file Containerfile --builder <docker-container builder> --target build-deconwolf \
#     --output type=local,dest=<folder> .            # then: sha256sum <folder>/opt/deconwolf/bin/dw
#   ... --build-arg DECONWOLF_SHA256=<that SHA-256> (with the other build arguments, every target) until the default is updated
#
# The source is the tag's archive, https://github.com/elgw/deconwolf/archive/refs/tags/v0.4.5.tar.gz, fetched by
# `ADD --checksum=sha256:<its SHA-256>` as the base images are pinned by digest (W-928: the archive's pax header names
# the commit ca062c49eedd6bf60fcd9804df467e64cfc3c113; CMakeLists.txt sets the version, so the build needs no Git). A
# remote ADD does not unpack: a RUN step does. `ADD --checksum` for an HTTP source needs a builder whose Dockerfile
# frontend is 1.6 or newer; the file names no `# syntax=` line, so the builder's own frontend parses it.
# psfmodels 0.3.3 (the `deconvolution` extra) has no wheel for Python 3.12 on Linux in the lock, so the CPU extras build
# it from its source distribution: the build stage has a C++ compiler.
#
# After a build, `src/python/scripts/image_check.py` checks the image (docs/workflow/broad-uger.md, "Checking a built
# image"): `uv pip check` may report only the one line that `piscis` requires `opencv-python`, which the lock leaves out
# on purpose (W-813, W-878), and `cv2`, `cellpose`, `piscis` and `stardist` must import together on the CPU.
#
# Every target has `ps` (procps) for Nextflow's trace, report and timeline (W-786), and none has Fiji, MATLAB, Java
# or Nextflow.

ARG CPU_EXTRAS="--extra ome --extra local-registration --extra checkpoint --extra anndata --extra multiview-stitcher --extra spatialdata --extra deconvolution"
ARG LEARNED_EXTRAS="--extra stardist --extra cellpose --extra spotiflow --extra piscis"
ARG CUDA_EXTRAS="--extra cuda"

FROM ghcr.io/astral-sh/uv:0.9.28@sha256:59240a65d6b57e6c507429b45f01b8f2c7c0bbeee0fb697c41a39c6a8e3a4cfb AS uv

FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337 AS build-deconwolf
# The apt sources of this stage only: one snapshot.debian.org timestamp (bookworm, bookworm-updates and bookworm-security,
# as the base image uses them), so that the compiler, CMake and the headers do not change when Debian publishes an update.
# The snapshot's Release files are accepted after their Valid-Until (`Check-Valid-Until: no`). The step stops unless both
# URIs were replaced. To move the snapshot, change the timestamp in both URIs and read the header above.
RUN sed -i -e 's#^URIs: http://deb.debian.org/debian$#URIs: http://snapshot.debian.org/archive/debian/20261124T000000Z#' \
      -e 's#^URIs: http://deb.debian.org/debian-security$#URIs: http://snapshot.debian.org/archive/debian-security/20261124T000000Z#' \
      -e '/^Signed-By:/a Check-Valid-Until: no' /etc/apt/sources.list.d/debian.sources \
    && test "$(grep -c '^URIs: http://snapshot.debian.org/archive/' /etc/apt/sources.list.d/debian.sources)" = 2
# bookworm's CMake 3.25 and the libraries deconwolf links: FFTW (single and double precision, with OpenMP), libtiff,
# GSL, libpng, the OpenCL ICD loader and headers.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake pkg-config ca-certificates \
      libfftw3-dev libtiff-dev libgsl-dev libpng-dev ocl-icd-opencl-dev opencl-headers \
    && rm -rf /var/lib/apt/lists/*
ADD --checksum=sha256:b9ff86f3ebb9d4b3c907d131ff018302a9eb0f5ebfb978cae29d40f75599ef67 https://github.com/elgw/deconwolf/archive/refs/tags/v0.4.5.tar.gz /src/deconwolf-0.4.5.tar.gz
RUN mkdir /src/deconwolf && tar -xzf /src/deconwolf-0.4.5.tar.gz -C /src/deconwolf --strip-components=1 \
    && rm /src/deconwolf-0.4.5.tar.gz
# SOURCE_DATE_EPOCH is the deconwolf v0.4.5 commit's time (ca062c49, 2025-03-12T08:25:06Z), which is the mtime of every entry
# of the tag's archive. GCC takes `__DATE__` and `__TIME__` from it, so the binary does not hold the build's day (W-936).
# It changes only with the deconwolf version.
ARG SOURCE_DATE_EPOCH=1741767906
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}
# -include inttypes.h: src/dw_imshift.c uses PRId64 without the header (W-915). The build folder is fixed, so the binary
# does not depend on the host's paths.
RUN cmake -S /src/deconwolf -B /src/build -DENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_FLAGS="-include inttypes.h" \
    && cmake --build /src/build -j 4 \
    && install -D -m 0755 /src/build/dw /opt/deconwolf/bin/dw \
    && install -D -m 0644 /src/build/src/trafo/libtrafo.so /opt/deconwolf/lib/libtrafo.so \
    && /src/build/dw --version | grep -q "deconwolf: '0.4.5'"

FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337 AS build
# A C++ compiler for psfmodels' source distribution (the deconvolution extra).
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LOCKED=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/starfinder \
    UV_CACHE_DIR=/root/.cache/uv
WORKDIR /src/python
COPY src/python/pyproject.toml src/python/uv.lock ./
ARG UV_LOCK_SHA256
RUN echo "${UV_LOCK_SHA256}  uv.lock" | sha256sum -c -

FROM build AS build-cpu
ARG CPU_EXTRAS
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-default-groups --group workflow $CPU_EXTRAS
COPY src/python/ ./
# hatch-vcs honours SETUPTOOLS_SCM_PRETEND_VERSION, not the _FOR_STARFINDER form (W-734).
ARG STARFINDER_VERSION
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$STARFINDER_VERSION
ARG STARFINDER_REVISION
ARG STARFINDER_DIRTY
RUN echo "$STARFINDER_REVISION" | grep -Eq '^[0-9a-f]{40}$' && { [ "$STARFINDER_DIRTY" = true ] || [ "$STARFINDER_DIRTY" = false ]; } \
    && printf 'COMMIT = "%s"\nDIRTY = %s\n' "$STARFINDER_REVISION" "$([ "$STARFINDER_DIRTY" = true ] && echo True || echo False)" \
       > starfinder/_build_info.py
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group workflow --no-editable $CPU_EXTRAS

FROM build AS build-learned
ARG CPU_EXTRAS
ARG LEARNED_EXTRAS
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-default-groups --group workflow $CPU_EXTRAS $LEARNED_EXTRAS
COPY src/python/ ./
ARG STARFINDER_VERSION
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$STARFINDER_VERSION
ARG STARFINDER_REVISION
ARG STARFINDER_DIRTY
RUN echo "$STARFINDER_REVISION" | grep -Eq '^[0-9a-f]{40}$' && { [ "$STARFINDER_DIRTY" = true ] || [ "$STARFINDER_DIRTY" = false ]; } \
    && printf 'COMMIT = "%s"\nDIRTY = %s\n' "$STARFINDER_REVISION" "$([ "$STARFINDER_DIRTY" = true ] && echo True || echo False)" \
       > starfinder/_build_info.py
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group workflow --no-editable $CPU_EXTRAS $LEARNED_EXTRAS
# The lock holds only opencv-python-headless; the slim base has no X11 or OpenGL libraries, so this import stops the build
# if a GUI OpenCV ever returns.
RUN /opt/starfinder/bin/python -c "import cv2, cellpose, piscis"

FROM build AS build-learned-cuda
ARG CPU_EXTRAS
ARG CUDA_EXTRAS
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-default-groups --group workflow $CPU_EXTRAS $CUDA_EXTRAS
COPY src/python/ ./
ARG STARFINDER_VERSION
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$STARFINDER_VERSION
ARG STARFINDER_REVISION
ARG STARFINDER_DIRTY
RUN echo "$STARFINDER_REVISION" | grep -Eq '^[0-9a-f]{40}$' && { [ "$STARFINDER_DIRTY" = true ] || [ "$STARFINDER_DIRTY" = false ]; } \
    && printf 'COMMIT = "%s"\nDIRTY = %s\n' "$STARFINDER_REVISION" "$([ "$STARFINDER_DIRTY" = true ] && echo True || echo False)" \
       > starfinder/_build_info.py
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group workflow --no-editable $CPU_EXTRAS $CUDA_EXTRAS
# As in build-learned.
RUN /opt/starfinder/bin/python -c "import cv2, cellpose, piscis"

FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337 AS runtime
# bookworm's only procps release (main, updates and security on 2026-11-03); deconwolf's runtime libraries (FFTW single
# and double precision with OpenMP, libtiff, GSL, libpng, the OpenCL ICD loader, libgomp).
RUN apt-get update && apt-get install -y --no-install-recommends procps=2:4.0.2-3 \
      libfftw3-single3 libfftw3-double3 libtiff6 libgsl27 libpng16-16 ocl-icd-libopencl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# deconwolf and its libtrafo.so; the NVIDIA OpenCL driver, which `apptainer --nv` and `docker --gpus` bind from the host,
# is named for the ICD loader.
COPY --from=build-deconwolf /opt/deconwolf /opt/deconwolf
RUN mkdir -p /etc/OpenCL/vendors && echo libnvidia-opencl.so.1 > /etc/OpenCL/vendors/nvidia.icd \
    && ln -s /opt/deconwolf/bin/dw /usr/local/bin/dw \
    && echo /opt/deconwolf/lib > /etc/ld.so.conf.d/deconwolf.conf && ldconfig \
    && ! ldd /opt/deconwolf/bin/dw | grep -q "not found"
# The reproduced binary's SHA-256 (W-936; see the header). Pass another only for a stage whose recipe changed on purpose.
ARG DECONWOLF_SHA256=95bb636df6fc92a74f49350bc3bcb51ffcc73b84153c8da3b8be4cbe738edfce
RUN echo "${DECONWOLF_SHA256}  /opt/deconwolf/bin/dw" | sha256sum -c -
ENV PATH=/opt/starfinder/bin:$PATH OMP_NUM_THREADS=1 MPLBACKEND=Agg
ARG STARFINDER_VERSION
ARG STARFINDER_REVISION
ARG STARFINDER_DIRTY
ARG UV_LOCK_SHA256
LABEL org.opencontainers.image.title=starfinder \
      org.opencontainers.image.version=$STARFINDER_VERSION \
      org.opencontainers.image.revision=$STARFINDER_REVISION \
      org.starfinder.dirty=$STARFINDER_DIRTY \
      org.starfinder.uv-lock-sha256=$UV_LOCK_SHA256 \
      org.starfinder.deconwolf=0.4.5 \
      org.starfinder.deconwolf-sha256=$DECONWOLF_SHA256

FROM runtime AS learned
COPY --from=build-learned /opt/starfinder /opt/starfinder
LABEL org.starfinder.variant=learned

FROM runtime AS learned-cuda
COPY --from=build-learned-cuda /opt/starfinder /opt/starfinder
LABEL org.starfinder.variant=learned-cuda

FROM runtime AS cpu
COPY --from=build-cpu /opt/starfinder /opt/starfinder
LABEL org.starfinder.variant=cpu
