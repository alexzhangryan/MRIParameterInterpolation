# syntax=docker/dockerfile:1
#
# Environment for verify_varnet.py (and, later, for VarNet training) on CHTC.
#
# Design choices, each of which is load-bearing:
#   - pytorch-lightning 1.9.5: last 1.x release. The fastMRI repo is written
#     against the 1.x argparse API (Trainer.from_argparse_args, gpus=,
#     resume_from_checkpoint=), all removed in 2.0.
#   - torch 2.0.1+cu118: the newest torch the Lightning 1.9 compatibility
#     table covers, built with sm_90 so it runs natively on H100 as well as
#     A100 / L40 / V100 / RTX. Only the host driver is needed at runtime.
#   - h5py 3.3.0 against HDF5 1.10.6 from conda-forge: fastMRI issue #215
#     documents a memory leak with pip h5py (HDF5 >= 1.12.1) when converting
#     to torch tensors. Irrelevant for a short inference job, matters for
#     multi-day training, and this image is meant to serve both.
#     Holding HDF5 below 1.12.1 is the constraint that forces the other two
#     pins, so do not "modernise" them independently: conda-forge has no
#     hdf5 1.10.6 build for python 3.10 (1.10.6 predates py310 support), and
#     every h5py 3.6.0 build requires hdf5 >=1.12.1. python 3.9 + h5py 3.3.0
#     is the newest combination that still gets the old HDF5. Nothing in the
#     stack needs 3.10; torch 2.0.1 and Lightning 1.9.5 both support 3.9.
#     The hdf5 *build string* is pinned too, and must be: h5py 3.3.0 is linked
#     against an hdf5 built with the ROS3 VFD, but `hdf5=1.10.6` alone resolves
#     to build nompi_h3c11f04_101, which lacks it, and `import h5py` then dies
#     with "undefined symbol: H5Pset_fapl_ros3". Builds _1111 and _1114 carry
#     ROS3; _1114 is the newest and is what is pinned here.
#   - fastMRI pinned to the exact commit of the local checkout and installed
#     with --no-deps so pip cannot pull Lightning 2.x in through setup.cfg.
#
#   - linux/amd64 only. CHTC execute nodes are x86_64, the cu118 torch wheels
#     exist only for x86_64, and the pinned hdf5 build string above is a
#     linux-64 build (conda-forge's linux-aarch64 1.10.6 builds have different
#     hashes). This laptop is Apple Silicon, so without the platform pin
#     Docker builds an arm64 image and the conda solve fails with
#     "hdf5 1.10.6 nompi_h6a2412b_1114 does not exist". The FROM line pins it,
#     and `make build` passes --platform as well, so both routes work.
#
# Build and push (laptop, from this directory):
#   docker build --platform linux/amd64 -t <dockerhub_user>/fastmri-verify:2026-09 .
#   docker push  <dockerhub_user>/fastmri-verify:2026-09
# then set container_image in verify.sub. Use a dated tag, not :latest, so a
# rebuild never silently changes what a past result was produced with.

FROM --platform=linux/amd64 mambaorg/micromamba:1.5.8-jammy

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends git xz-utils ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

ARG FASTMRI_SHA=91f2df4711adbb6d643df1810f234e4abcf5881b
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# conda layer: python + the HDF5 stack that must not come from pip
RUN micromamba install -y -n base -c conda-forge \
        python=3.9 pip \
        "h5py=3.3.0" "hdf5=1.10.6=nompi_h6a2412b_1114" \
        numpy=1.23.5 scikit-image=0.19.3 pandas=1.5.3 pyyaml=6.0 \
 && micromamba clean -a -y

# pip layer: torch first from the CUDA index, then everything that depends on it
RUN pip install --no-cache-dir \
        torch==2.0.1 torchvision==0.15.2 \
        --index-url https://download.pytorch.org/whl/cu118 \
 && pip install --no-cache-dir \
        pytorch-lightning==1.9.5 torchmetrics==0.11.4 \
        runstats==2.0.0 wandb==0.18.7 requests tqdm pytest==7.4.4

# fastMRI at a pinned commit, installed non-editable, no dependency resolution
RUN git clone --quiet https://github.com/facebookresearch/fastMRI.git /opt/fastMRI \
 && git -C /opt/fastMRI checkout --quiet ${FASTMRI_SHA} \
 && pip install --no-cache-dir --no-deps /opt/fastMRI \
 && python -c "import fastmri, pytorch_lightning as pl, torch, h5py; \
               assert pl.__version__.startswith('1.'); \
               print('fastmri', fastmri.__version__, 'pl', pl.__version__, \
                     'torch', torch.__version__, 'h5py', h5py.__version__, 'hdf5', h5py.version.hdf5_version)"

# Recorded into every W&B run config and tier1_report.json by the harness.
ARG IMAGE_TAG=fastmri-verify:dev
ENV VERIFY_IMAGE=${IMAGE_TAG} \
    FASTMRI_REPO=/opt/fastMRI \
    PYTHONUNBUFFERED=1 \
    WANDB_SILENT=true

# CHTC runs the container as the submitting user with the job scratch
# directory as cwd, so nothing below relies on a writable home or on
# mambauser. Keep the entrypoint plain so `executable = run_verify.sh` works.
USER root
WORKDIR /work
