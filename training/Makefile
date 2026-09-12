# Makefile: driver for VarNet training on CHTC.
#
# Two kinds of targets, run in two places:
#   laptop (needs Docker):        make build push smoke job-smoke
#   ap2001.chtc.wisc.edu:         make submit-model1 submit-model2 resume status logs
#
#   make help            every target
#
# Requires GNU make, Docker. On Windows run it from Git Bash or WSL.
#
# CHTC is x86_64, so every docker build and run here passes --platform
# $(PLATFORM), default linux/amd64. On an Apple Silicon laptop that runs under
# emulation: slow, but correct, and fine for tiny synthetic data. Override it
# when you are building somewhere that matches the host, e.g. on a linux/arm64
# box you only ever smoke-test on:
#     make smoke PLATFORM=linux/arm64
# Anything you `make push` for CHTC must stay linux/amd64.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# --- knobs ------------------------------------------------------------------
# Comments go above the assignments, never after them: make keeps the trailing
# whitespace, which turns `ifeq ($(GPU),1)` into a silent false.
#
# IMAGE: use a dated tag (<user>/fastmri-train:2026-09) for anything you push.
IMAGE    ?= fastmri-train:local
PLATFORM ?= linux/amd64
# model preset for `make submit` / `make resume`: model1 | model2
MODEL    ?= model1
# extra name=value macro overrides for condor_submit, e.g.
#   make submit-model2 ARGS='max_epochs=2 extra_args="--limit_train_batches 20"'
ARGS     ?=
# checkpoint for `make resume`
CKPT     ?=
# synthetic volumes for the smoke tests
VOLUMES  ?= 3
SLICES   ?= 4
# GPU: 1 uses the host GPU, 0 forces CPU. Auto-detected from nvidia-smi.
GPU      ?= $(if $(shell nvidia-smi -L 2>/dev/null),1,0)
SHM      ?= 8g
# extra args appended to train_wandb.py in the smoke targets
EXTRA    ?=

# Tiny architecture for the smoke tests: the point is exercising the data
# path, the checkpoint/resume logic and the job executable, not the model.
SMOKE_ARCH ?= --num_cascades 2 --chans 4 --sens_chans 4 --pools 2 --sens_pools 2
SMOKE_RUN  ?= --limit_train_batches 4 --limit_val_batches 2 --num_workers 0 --no_wandb

# --- host/docker plumbing ---------------------------------------------------
# The repo root is mounted (not just this directory) so the smoke targets can
# reuse ../verification/make_synthetic_val.py without copying it.
#
# Windows: Git Bash mangles both the host path and the container-side path.
# `pwd -W` gives the C:/... form Docker Desktop wants (plain $(CURDIR) yields
# /c/... , which Docker rejects as a bind mount source) and MSYS_NO_PATHCONV=1
# stops Git Bash rewriting `-w /work/training` into a C:/Program Files/... path.
# --user is dropped there too: Docker Desktop already maps the bind mount as
# root, and a Git Bash uid (197609) does not exist inside the image.
ifeq ($(OS),Windows_NT)
  REPO     := $(shell cd .. && pwd -W 2>/dev/null || pwd)
  DOCKER   := MSYS_NO_PATHCONV=1 docker
  USERFLAG :=
else
  REPO     := $(abspath $(CURDIR)/..)
  DOCKER   := docker
  USERFLAG := --user $(shell id -u):$(shell id -g)
endif
ifeq ($(GPU),1)
  GPUFLAG := --gpus all
  GPUS    := 1
else
  GPUFLAG :=
  GPUS    := 0
endif
# HOME and the cache dirs go to /tmp: the CHTC container has no writable home
# either, so depending on one is a bug that would only show up on the cluster.
ENVFLAGS := -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache -e MPLCONFIGDIR=/tmp/.cache/mpl \
            -e PYTHONUNBUFFERED=1 -e WANDB_MODE=offline
RUN      := $(DOCKER) run --rm --platform $(PLATFORM) --shm-size=$(SHM) $(GPUFLAG) $(USERFLAG) $(ENVFLAGS) \
            -v "$(REPO)":/work -w /work/training $(IMAGE)

STAMP := .make/image.$(subst :,_,$(subst /,_,$(IMAGE)))
SYNTH := synthetic/knee

# ============================================================================

.PHONY: help
help:
	@echo "VarNet training on CHTC (IMAGE=$(IMAGE))"
	@echo
	@echo "  LAPTOP (Docker):"
	@echo "  make build           build the image for $(PLATFORM) (cached)"
	@echo "  make push IMAGE=<user>/fastmri-train:2026-09   publish for CHTC"
	@echo "  make smoke           train_wandb.py on synthetic data, then resume from last.ckpt"
	@echo "  make smoke-wandb     same, with the W&B logger attached (offline): covers log_image"
	@echo "  make job-smoke       run_train.sh the way HTCondor will, twice (fresh + evicted)"
	@echo "  make job-evict       SIGTERM a running job: eviction cleanup + last.ckpt survive"
	@echo "  make shell           interactive shell in the image, repo mounted at /work"
	@echo
	@echo "  ACCESS POINT (ap2001.chtc.wisc.edu, WANDB_API_KEY exported):"
	@echo "  make subset-val      one-off: val subset from the staged tarball (README 3b, step 1)"
	@echo "  make subset-train    one-off: train subset streamed from NYU into /staging (step 2)"
	@echo "  make submit-model1   train on acceleration 4 only"
	@echo "  make submit-model2   train on accelerations 2 4 6 8, one drawn per sample"
	@echo "  make submit MODEL=model2 ARGS='max_epochs=2'"
	@echo "  make resume MODEL=model1 CKPT=runs/<model>/<Cluster>/checkpoints/last.ckpt"
	@echo "  make status          condor_q for your jobs"
	@echo "  make logs            tail the newest job .out"
	@echo
	@echo "  make clean           remove smoke artefacts;  make distclean also drops logs/ runs/ and the image"
	@echo
	@echo "knobs: IMAGE= PLATFORM= MODEL= ARGS= CKPT= GPU= SHM= VOLUMES= SLICES= EXTRA='...'"
	@echo "       PLATFORM defaults to linux/amd64 (what CHTC runs); pass e.g."
	@echo "       PLATFORM=linux/arm64 to build natively for local smoke tests only."

# --- image ------------------------------------------------------------------
.PHONY: build rebuild push
build: $(STAMP)

$(STAMP): Dockerfile
	@mkdir -p .make
	$(DOCKER) build --platform $(PLATFORM) --build-arg IMAGE_TAG=$(IMAGE) -t $(IMAGE) .
	@touch $@

rebuild:
	@mkdir -p .make
	$(DOCKER) build --no-cache --platform $(PLATFORM) --build-arg IMAGE_TAG=$(IMAGE) -t $(IMAGE) .
	@touch $(STAMP)

push: build
	@case "$(IMAGE)" in */*:*) ;; *) \
	  echo "IMAGE must be <dockerhub_user>/<name>:<dated-tag>, got '$(IMAGE)'" >&2; exit 1;; esac
	@case "$(IMAGE)" in *:latest) \
	  echo "refusing :latest -- a rebuild would silently change what a checkpoint was produced with" >&2; \
	  exit 1;; esac
	$(DOCKER) push $(IMAGE)
	@echo
	@echo "now set in train.sub:   image = docker://$(IMAGE)"

# --- line endings -----------------------------------------------------------
# These files execute on Linux; a stray CRLF makes bash fail on "\r".
#
# The Makefile and Dockerfile are checked too, because they fail differently
# and much less obviously: make keeps the \r in every variable, so a CRLF
# Makefile turns `PLATFORM ?= linux/amd64` into "linux/amd64\r" and every
# `docker build --platform ...` dies on an invalid platform. .gitattributes
# pins eol=lf so a Windows checkout cannot reintroduce it.
.PHONY: check-eol
check-eol:
	@bad=""; \
	for f in Makefile Dockerfile run_train.sh submit.sh make_subset.sh train.sub subset_val.sub subset_train.sub train_wandb.py; do \
	  if LC_ALL=C grep -qU "$$(printf '\r')" "$$f" 2>/dev/null; then bad="$$bad $$f"; fi; \
	done; \
	if [ -n "$$bad" ]; then \
	  echo "CRLF line endings found in:$$bad  (fix: dos2unix, or sed -i 's/\r$$//')" >&2; exit 1; \
	fi

# --- synthetic data ---------------------------------------------------------
# Phantoms with the real on-disk layout, one directory per split. Numbers
# trained on this are meaningless; it only exercises the pipeline.
.PHONY: synthetic
synthetic: $(SYNTH)/multicoil_val

$(SYNTH)/multicoil_val: ../verification/make_synthetic_val.py | build
	$(RUN) python ../verification/make_synthetic_val.py --out $(SYNTH)/multicoil_train \
	    --volumes $(VOLUMES) --slices $(SLICES) --seed 1
	$(RUN) python ../verification/make_synthetic_val.py --out $(SYNTH)/multicoil_val \
	    --volumes 2 --slices $(SLICES) --seed 2
	@touch $@

knee_multicoil_train_synthetic.tar knee_multicoil_val_synthetic.tar: $(SYNTH)/multicoil_val
	tar -cf knee_multicoil_train_synthetic.tar -C $(SYNTH) multicoil_train
	tar -cf knee_multicoil_val_synthetic.tar   -C $(SYNTH) multicoil_val

# --- smoke: the driver ------------------------------------------------------
# Epoch 1 from scratch, then epoch 2 must start from output/checkpoints/last.ckpt.
.PHONY: smoke
smoke: build check-eol synthetic
	rm -rf output
	$(RUN) python train_wandb.py --data_path $(SYNTH) --default_root_dir output \
	    --run_name smoke --accelerations 2 4 6 8 --center_fractions 0.16 0.08 0.0533 0.04 \
	    --gpus $(GPUS) --max_epochs 1 $(SMOKE_ARCH) $(SMOKE_RUN) $(EXTRA)
	@test -s output/checkpoints/last.ckpt || { echo "FAILED: no output/checkpoints/last.ckpt after epoch 1" >&2; exit 1; }
	@echo "--- epoch 1 done, checkpoint written. Re-running with --max_epochs 2: must resume ---"
	$(RUN) python train_wandb.py --data_path $(SYNTH) --default_root_dir output \
	    --run_name smoke --accelerations 2 4 6 8 --center_fractions 0.16 0.08 0.0533 0.04 \
	    --gpus $(GPUS) --max_epochs 2 $(SMOKE_ARCH) $(SMOKE_RUN) $(EXTRA) 2>&1 | tee output/resume.log
	@grep -q "resume from: output/checkpoints/last.ckpt" output/resume.log || { \
	  echo "FAILED: second run did not resume from last.ckpt" >&2; exit 1; }
	@echo
	@echo "smoke passed: training runs, last.ckpt is written, and a rerun resumes from it."

# The W&B logger is the one the cluster actually uses, and SMOKE_RUN above
# disables it -- so `make smoke` exercises TensorBoardLogger, whose experiment
# object is the only one that really has add_image. Every W&B-specific path was
# therefore untested, which is how an AttributeError in MriModule.log_image
# reached CHTC and killed three jobs a full epoch in. This target is the same
# smoke with W&B attached; ENVFLAGS already pins WANDB_MODE=offline, so it needs
# no key and uploads nothing. Target-specific variables reach prerequisites, so
# dropping --no_wandb from SMOKE_RUN here is the whole difference.
.PHONY: smoke-wandb
smoke-wandb: SMOKE_RUN := --limit_train_batches 4 --limit_val_batches 2 --num_workers 0
smoke-wandb: smoke
	@ls output/wandb/*/files/media/images/* >/dev/null 2>&1 || { \
	  echo "FAILED: no media under output/wandb/*/files/media/images: validation never" >&2; \
	  echo "        called log_image, so this proved nothing about the W&B path." >&2; \
	  exit 1; }
	@echo
	@echo "smoke-wandb passed: validation images reached the W&B logger, not just TensorBoard."

# --- smoke: the job executable ----------------------------------------------
# run_train.sh does what the driver does not: tarball extraction, split
# discovery, the output/ contract, checkpoint seeding. Run it twice in the same
# scratch dir: the second run sees output/ exactly as HTCondor would deliver it
# after an eviction and must resume rather than restart.
.PHONY: job-smoke
job-smoke: build check-eol knee_multicoil_train_synthetic.tar
	rm -rf jobtest && mkdir -p jobtest
	cp train_wandb.py run_train.sh knee_multicoil_*_synthetic.tar jobtest/
	$(DOCKER) run --rm --platform $(PLATFORM) --shm-size=$(SHM) $(GPUFLAG) $(USERFLAG) $(ENVFLAGS) -e KEEP_TARBALL=1 \
	    -v "$(REPO)":/work -w /work/training/jobtest $(IMAGE) \
	    bash run_train.sh --run_name job-smoke --accelerations 4 --center_fractions 0.08 \
	        --gpus $(GPUS) --max_epochs 1 $(SMOKE_ARCH) $(SMOKE_RUN) $(EXTRA)
	@test -s jobtest/output/checkpoints/last.ckpt || { echo "FAILED: run_train.sh produced no output/checkpoints/last.ckpt" >&2; exit 1; }
	@echo "--- first job done. Second run in the same sandbox (= after an eviction): must resume ---"
	rm -rf jobtest/data
	$(DOCKER) run --rm --platform $(PLATFORM) --shm-size=$(SHM) $(GPUFLAG) $(USERFLAG) $(ENVFLAGS) -e KEEP_TARBALL=1 \
	    -v "$(REPO)":/work -w /work/training/jobtest $(IMAGE) \
	    bash run_train.sh --run_name job-smoke --accelerations 4 --center_fractions 0.08 \
	        --gpus $(GPUS) --max_epochs 2 $(SMOKE_ARCH) $(SMOKE_RUN) $(EXTRA) 2>&1 | tee jobtest/resume.log
	@grep -q "resume from: output/checkpoints/last.ckpt" jobtest/resume.log || { \
	  echo "FAILED: evicted-job rerun did not resume from last.ckpt" >&2; exit 1; }
	@echo
	@echo "job-smoke passed: extraction, split discovery, output/ contract, and resume-after-eviction."

# --- smoke: the eviction path ------------------------------------------------
# HTCondor vacates a multi-day GPU Lab job by sending SIGTERM to the executable
# and SIGKILLing the whole job after MachineMaxVacateTime. Neither smoke target
# above goes near that: both let training finish on its own. This one starts
# run_train.sh with W&B attached, SIGTERMs it mid-training the way the startd
# does, and checks the script actually stops and finishes its cleanup.
#
# It exists because the eviction path was never exercised and turned out to rest
# on a wrong assumption: bash's `wait` returns without reaping the child as soon
# as a trapped signal arrives, so the script was exiting while python was still
# writing into output/. The obvious "just wait longer" fix is worse -- if python
# ever does trap SIGTERM, an unbounded wait runs past HTCondor's SIGKILL and the
# symlink sweep never happens, re-holding the job on output transfer. This test
# pins down the behaviour either way. VACATE_GRACE is shortened here for speed.
.PHONY: job-evict
job-evict: build check-eol knee_multicoil_train_synthetic.tar
	rm -rf evicttest && mkdir -p evicttest
	cp train_wandb.py run_train.sh knee_multicoil_*_synthetic.tar evicttest/
	-@$(DOCKER) rm -f fastmri-evict >/dev/null 2>&1 || true
	$(DOCKER) run -d --name fastmri-evict --platform $(PLATFORM) --shm-size=$(SHM) $(GPUFLAG) $(USERFLAG) \
	    $(ENVFLAGS) -e KEEP_TARBALL=1 -e VACATE_GRACE=5 \
	    -v "$(REPO)":/work -w /work/training/evicttest $(IMAGE) \
	    bash run_train.sh --run_name job-evict --accelerations 4 --center_fractions 0.08 \
	        --gpus $(GPUS) --max_epochs 10000 $(SMOKE_ARCH) --num_workers 0 $(EXTRA)
	@echo "--- waiting for training to start, then SIGTERM (what HTCondor sends on a vacate) ---"
	@for i in $$(seq 1 150); do \
	   $(DOCKER) logs fastmri-evict 2>&1 | tr '\r' '\n' | grep -q "^Epoch 1" && break; sleep 2; done
	@$(DOCKER) logs fastmri-evict 2>&1 | tr '\r' '\n' | grep -q "^Epoch 1" || { \
	   echo "FAILED: training never started, nothing to evict" >&2; \
	   $(DOCKER) logs fastmri-evict 2>&1 | tail -20 >&2; $(DOCKER) rm -f fastmri-evict >/dev/null; exit 1; }
	@$(DOCKER) kill -s TERM fastmri-evict >/dev/null
	@t0=$$(date +%s); \
	 while [ "$$($(DOCKER) inspect -f '{{.State.Running}}' fastmri-evict)" = "true" ]; do \
	   if [ $$(( $$(date +%s) - t0 )) -gt 90 ]; then \
	     echo "FAILED: run_train.sh did not exit within 90s of SIGTERM; on CHTC this is the" >&2; \
	     echo "        SIGKILL case, where the cleanup below never runs." >&2; \
	     $(DOCKER) rm -f fastmri-evict >/dev/null; exit 1; \
	   fi; sleep 1; \
	 done; echo "exited $$(( $$(date +%s) - t0 ))s after SIGTERM"
	@$(DOCKER) logs fastmri-evict 2>&1 | tr '\r' '\n' | grep -E "^\[run_train\] (SIGTERM|python|exit code)"
	@$(DOCKER) logs fastmri-evict 2>&1 | tr '\r' '\n' | grep -q "SIGTERM received" || { \
	   echo "FAILED: the trap never fired" >&2; $(DOCKER) rm -f fastmri-evict >/dev/null; exit 1; }
	@$(DOCKER) rm -f fastmri-evict >/dev/null
	@l=$$(find evicttest/output -type l | tr '\n' ' '); \
	 test -z "$$l" || { echo "FAILED: symlinks survived the cleanup, HTCondor would hold the" >&2; \
	   echo "        job on output transfer: $$l" >&2; exit 1; }
	@test -s evicttest/output/checkpoints/last.ckpt || { \
	   echo "FAILED: no last.ckpt, the evicted run would restart from scratch" >&2; exit 1; }
	@echo
	@echo "job-evict passed: SIGTERM stops the job, the symlink sweep runs, last.ckpt survives."

# --- CHTC access point ------------------------------------------------------
.PHONY: submit submit-model1 submit-model2 resume status logs subset-val subset-train
# README section 3b: build the subset tarballs that fit under the /staging
# quota. subset-val brings ~N_VAL GB back to this directory; subset-train
# needs FASTMRI_TRAIN_URL (from .env) and writes straight into /staging.
N_VAL           ?= 20
TRAIN_PREFIX_GB ?= 65
subset-val:
	@mkdir -p logs
	condor_submit subset_val.sub n_val=$(N_VAL)

subset-train:
	@mkdir -p logs
	@set -a; [ -f .env ] && . ./.env; set +a; \
	test -n "$$FASTMRI_TRAIN_URL" || { echo "FASTMRI_TRAIN_URL is empty: put the NYU presigned URL in .env (see .env.example)" >&2; exit 1; }; \
	condor_submit subset_train.sub train_prefix_gb=$(TRAIN_PREFIX_GB)

submit-model1:
	./submit.sh model1 $(ARGS)

submit-model2:
	./submit.sh model2 $(ARGS)

submit:
	./submit.sh $(MODEL) $(ARGS)

resume:
	@test -n "$(CKPT)" || { echo "usage: make resume MODEL=model1 CKPT=runs/<model>/<Cluster>/checkpoints/last.ckpt" >&2; exit 1; }
	./submit.sh $(MODEL) resume=$(CKPT) $(ARGS)

status:
	condor_q -nobatch

logs:
	@f=$$(ls -t logs/*.out 2>/dev/null | head -1); \
	test -n "$$f" || { echo "no logs/*.out yet" >&2; exit 1; }; \
	echo "== $$f"; tail -f "$$f"

# --- misc -------------------------------------------------------------------
.PHONY: shell
shell: build
	$(DOCKER) run --rm -it --platform $(PLATFORM) --shm-size=$(SHM) $(GPUFLAG) $(USERFLAG) $(ENVFLAGS) \
	    -v "$(REPO)":/work -w /work/training $(IMAGE) bash

.PHONY: clean distclean
clean:
	rm -rf output synthetic jobtest evicttest knee_multicoil_*_synthetic.tar

distclean: clean
	rm -rf .make logs runs
	-$(DOCKER) rmi $(IMAGE)
