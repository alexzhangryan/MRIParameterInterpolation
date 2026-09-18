# syntax=docker/dockerfile:1
#
# Environment for VarNet training (train_wandb.py) on CHTC. Same pins as
# verification/Dockerfile, for the same reasons; the two images are kept
# separate so a training rebuild can never change what a verification result
# was produced with.
#
# Design choices, each of which is load-bearing:
#   - pytorch-lightning 1.9.5: last 1.x release. The fastMRI repo is written
#     against the 1.x argparse API (Trainer.from_argparse_args, gpus=,
#     validation_epoch_end), all removed in 2.0.
#   - torch 2.0.1+cu118: the newest torch the Lightning 1.9 compatibility
#     table covers, built with sm_90 so it runs natively on H100 as well as
#     A100 / L40 / V100 / RTX. Only the host driver is needed at runtime.
#   - h5py 3.3.0 against HDF5 1.10.6 from conda-forge: fastMRI issue #215
#     documents a memory leak with pip h5py (HDF5 >= 1.12.1) when converting
#     to torch tensors. It matters for a multi-day training job. Holding HDF5
#     below 1.12.1 forces python 3.9 (no py310 build of hdf5 1.10.6) and the
#     hdf5 build string must carry the ROS3 VFD or `import h5py` dies with
#     "undefined symbol: H5Pset_fapl_ros3"; _1114 is the newest that does.
#   - fastMRI pinned to the exact commit of the fastMRI/ submodule and
#     installed with --no-deps so pip cannot pull Lightning 2.x in through
#     setup.cfg. The submodule itself is never copied into the image.
#
# Build and push (laptop, from this directory). This Mac is Apple Silicon and
# CHTC is x86_64; the cu118 torch wheels and the pinned hdf5 build string
# only exist for x86_64, so the platform is pinned in the FROM line below and
# `make build` / `make push` pass --platform as well.
#   docker build --platform linux/amd64 -t <dockerhub_user>/fastmri-train:2026-09 .
#   docker push  <dockerhub_user>/fastmri-train:2026-09
# then set container_image in train.sub. Use a dated tag, not :latest, so a
# rebuild never silently changes what a checkpoint was produced with.

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
# wandb >= 0.26: 0.18.7 rejected W&B's current 86-character API keys before
# contacting the server ("API key must be 40 characters long"), which the
# W&B preflight in train_wandb.py turns into a held job. Bumped 2026-09-18.
 && pip install --no-cache-dir \
        pytorch-lightning==1.9.5 torchmetrics==0.11.4 tensorboard==2.14.1 \
        runstats==2.0.0 wandb==0.26.1 requests tqdm

# fastMRI at a pinned commit, installed non-editable, no dependency resolution
RUN git clone --quiet https://github.com/facebookresearch/fastMRI.git /opt/fastMRI \
 && git -C /opt/fastMRI checkout --quiet ${FASTMRI_SHA} \
 && pip install --no-cache-dir --no-deps /opt/fastMRI \
 && python -c "import fastmri, pytorch_lightning as pl, torch, h5py, wandb; \
               from pytorch_lightning.loggers import WandbLogger; \
               assert pl.__version__.startswith('1.'); \
               print('fastmri', fastmri.__version__, 'pl', pl.__version__, \
                     'torch', torch.__version__, 'h5py', h5py.__version__, \
                     'hdf5', h5py.version.hdf5_version, 'wandb', wandb.__version__)"

# Recorded into every W&B run config by run_train.sh / train_wandb.py.
ARG IMAGE_TAG=fastmri-train:dev
ENV TRAIN_IMAGE=${IMAGE_TAG} \
    FASTMRI_REPO=/opt/fastMRI \
    PYTHONUNBUFFERED=1 \
    WANDB_SILENT=true

# CHTC runs the container as the submitting user with the job scratch
# directory as cwd, so nothing below relies on a writable home or on
# mambauser. Keep the entrypoint plain so `executable = run_train.sh` works.
USER root
WORKDIR /work
