# The workflow image from uv.lock (design/specs/orchestration-engines.md, "The image").
#
# Three targets from the same lock: `cpu`, the default image (the CPU extras and the `workflow` group);
# `learned`, which adds the learned extras on CPU for `segment` and the learned detectors (W-646); and
# `learned-cuda`, which adds the lock's `cuda` extra instead: the same learned backends with torch and torchvision
# from the PyTorch cu126 index and TensorFlow with its CUDA libraries (W-792). The CUDA runtime comes from NVIDIA's
# wheels and the driver from the host (`apptainer exec --nv`, `docker run --gpus`).
# The build context has no .git: the version is `git describe`'s on the build host, passed in, and the
# revision and the lock's SHA-256 are passed in for the labels; the build stops if the lock differs.
#
#   docker buildx build --builder <docker-container builder> --target cpu \
#     --build-arg STARFINDER_VERSION=<version> --build-arg STARFINDER_REVISION=<commit> \
#     --build-arg UV_LOCK_SHA256=<sha256 of src/python/uv.lock> \
#     --output type=oci,dest=starfinder-cpu-<version>.oci.tar .
#
# Every target has `ps` (procps) for Nextflow's trace, report and timeline (W-786), and none has Fiji, MATLAB, Java
# or Nextflow.

ARG CPU_EXTRAS="--extra ome --extra local-registration --extra checkpoint --extra anndata --extra multiview-stitcher --extra spatialdata"
ARG LEARNED_EXTRAS="--extra stardist --extra cellpose --extra spotiflow --extra piscis"
ARG CUDA_EXTRAS="--extra cuda"

FROM ghcr.io/astral-sh/uv:0.9.28@sha256:59240a65d6b57e6c507429b45f01b8f2c7c0bbeee0fb697c41a39c6a8e3a4cfb AS uv

FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337 AS build
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
# piscis locks opencv-python and cellpose opencv-python-headless, and both write cv2/: the copy that remains depends on
# the order of installation. The GUI copy needs X11, OpenGL and GLib libraries the slim base lacks, so the headless
# copy, the same locked wheel, is installed again last; no learned package uses OpenCV's GUI.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group workflow --no-editable \
      --reinstall-package opencv-python-headless $CPU_EXTRAS $LEARNED_EXTRAS

FROM build AS build-learned-cuda
ARG CPU_EXTRAS
ARG CUDA_EXTRAS
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-default-groups --group workflow $CPU_EXTRAS $CUDA_EXTRAS
COPY src/python/ ./
ARG STARFINDER_VERSION
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$STARFINDER_VERSION
# The headless OpenCV last, as in build-learned.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group workflow --no-editable \
      --reinstall-package opencv-python-headless $CPU_EXTRAS $CUDA_EXTRAS

FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337 AS runtime
# bookworm's only procps release (main, updates and security on 2026-11-03).
RUN apt-get update && apt-get install -y --no-install-recommends procps=2:4.0.2-3 && rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/starfinder/bin:$PATH OMP_NUM_THREADS=1 MPLBACKEND=Agg
ARG STARFINDER_VERSION
ARG STARFINDER_REVISION
ARG UV_LOCK_SHA256
LABEL org.opencontainers.image.title=starfinder \
      org.opencontainers.image.version=$STARFINDER_VERSION \
      org.opencontainers.image.revision=$STARFINDER_REVISION \
      org.starfinder.uv-lock-sha256=$UV_LOCK_SHA256

FROM runtime AS learned
COPY --from=build-learned /opt/starfinder /opt/starfinder
LABEL org.starfinder.variant=learned

FROM runtime AS learned-cuda
COPY --from=build-learned-cuda /opt/starfinder /opt/starfinder
LABEL org.starfinder.variant=learned-cuda

FROM runtime AS cpu
COPY --from=build-cpu /opt/starfinder /opt/starfinder
LABEL org.starfinder.variant=cpu
