# Makefile: local (laptop) driver for the verification harness.
#
# Everything here runs in the same Docker image CHTC runs, so a green local run
# means the image, the harness, and the job executable all work before a single
# HTCondor slot is used. Nothing here touches /staging, condor, or the real
# 94 GB dataset -- see README.md sections 3-5 for those.
#
#   make local-run      full smoke test from scratch: build, fake data, tier0,
#                       tier1, job executable. This is the one to run.
#   make help           every target
#
# Requires: GNU make, Docker. On Windows run it from Git Bash or WSL.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# --- knobs ------------------------------------------------------------------
# Comments go above the assignments, never after them: make keeps the trailing
# whitespace, which turns `ifeq ($(PYTEST),1)` into a silent false.
#
# IMAGE: use a dated tag (fastmri-verify:2026-09) for anything you push.
IMAGE    ?= fastmri-verify:local
# PLATFORM: CHTC is x86_64 and the torch cu118 wheels / pinned hdf5 build only
# exist there. On an Apple Silicon laptop this makes docker build and run use
# emulation (slow, but correct); without it the conda solve fails outright.
PLATFORM ?= linux/amd64
RESULTS  ?= results
SYNTH    ?= synthetic
# synthetic volumes, and slices per volume
VOLUMES  ?= 3
SLICES   ?= 4
# paired elementwise, as in fastmri.data.subsample.MaskFunc
ACCEL    ?= 4 8
CF       ?= 0.08 0.04
# random is the fastMRI knee convention and the only one the reference values
# in verify_varnet.py are valid for (see README "Mask family notes").
MASK_TYPE ?= random
# PYTEST=1: tier0 also runs the fastMRI repo test suite
PYTEST   ?= 1
# GPU: 1 uses the host GPU, 0 forces CPU. Auto-detected from nvidia-smi because
# the difference is not cosmetic -- the real Tier 1 run is hours on a GPU and
# days on CPU, and a silent CPU fallback is easy not to notice.
# (`nvidia-smi -L` rather than `command -v`: GnuWin32 make 3.81 cannot run a
# shell builtin in $(shell ...) and dies with a CreateProcess error.)
GPU      ?= $(if $(shell nvidia-smi -L 2>/dev/null),1,0)
# extra args appended to verify_varnet.py
EXTRA    ?=

# Tiny architecture for the smoke test. The released checkpoint's real
# architecture is the default in verify_varnet.py; we override it here only
# because --random_init means the weights are meaningless anyway and a small
# model keeps the CPU run to minutes.
SMOKE_ARCH ?= --num_cascades 2 --chans 4 --sens_chans 4 --pools 2 --sens_pools 2

# --- host/docker plumbing ---------------------------------------------------
# Git Bash mangles both $PWD and the container-side path; pwd -W gives the
# Windows form Docker Desktop wants and MSYS_NO_PATHCONV leaves /work alone.
ifeq ($(OS),Windows_NT)
  HOSTPWD  := $(shell pwd -W 2>/dev/null || pwd)
  DOCKER   := MSYS_NO_PATHCONV=1 docker
  USERFLAG :=
else
  HOSTPWD  := $(CURDIR)
  DOCKER   := docker
  USERFLAG := --user $(shell id -u):$(shell id -g)
endif

ifeq ($(GPU),1)
  GPUFLAG := --gpus all
  CPUFLAG :=
else
  GPUFLAG :=
  CPUFLAG := --cpu
endif

ifeq ($(PYTEST),1)
  PYTESTFLAG := --run_pytest
else
  PYTESTFLAG :=
endif

# HOME and the cache dirs go to /tmp: the CHTC container has no writable home
# either, so depending on one is a bug that would only show up on the cluster.
ENVFLAGS := -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache -e MPLCONFIGDIR=/tmp/.cache/mpl \
            -e PYTHONUNBUFFERED=1
RUN      := $(DOCKER) run --rm --platform $(PLATFORM) $(SHMFLAG) $(GPUFLAG) $(USERFLAG) $(ENVFLAGS) \
            -v "$(HOSTPWD)":/work -w /work $(IMAGE)

# PyTorch DataLoader workers hand samples to the parent process through
# /dev/shm, and Docker gives a container only 64 MB of it by default. One
# fastMRI slice is 15 x 640 x 368 complex64 = 28 MB, so the default
# --num_workers 4 prefetching a few of those exceeds 64 MB immediately and the
# workers die with "Bus error ... insufficient shared memory". tier1-smoke
# never hit this because it passes --num_workers 0. Whether the CHTC container
# has a larger /dev/shm is worth checking there before assuming it is fine.
# Weights & Biases. Every smoke target passes --no_wandb literally and stays
# silent; this knob is for `make tier1`, the run worth a permanent record.
# WANDB=1 (default) forwards the credentials from .env and lets the script
# choose: online when WANDB_API_KEY arrives, offline otherwise, in which case
# it prints the `wandb sync` command. Force it with WANDB_MODE=offline in the
# environment. WANDB=0 turns logging off entirely.
WANDB ?= 1
ifeq ($(WANDB),1)
  WANDBFLAG :=
  WANDBENV  := -e WANDB_API_KEY -e WANDB_ENTITY -e WANDB_PROJECT -e WANDB_MODE
else
  WANDBFLAG := --no_wandb
  WANDBENV  := -e WANDB_MODE=offline
endif

SHM      ?= 8g
SHMFLAG  := --shm-size=$(SHM)

STAMP := .make/image.$(subst :,_,$(subst /,_,$(IMAGE)))

# ============================================================================

.PHONY: help
help:
	@echo "local verification targets (IMAGE=$(IMAGE))"
	@echo
	@echo "  make local-run       everything from 0: build, synthetic data, tier0,"
	@echo "                       tier1 smoke, job-executable smoke"
	@echo "  make build           build the Docker image (cached; no-op if unchanged)"
	@echo "  make rebuild         build with --no-cache"
	@echo "  make synthetic       generate $(SYNTH)/multicoil_val ($(VOLUMES) vols x $(SLICES) slices)"
	@echo "  make tier0           environment + metric invariants (+ fastMRI pytest)"
	@echo "  make tier1-smoke     tier1 on synthetic data with a random model"
	@echo "  make job-smoke       run run_verify.sh itself, the way HTCondor will"
	@echo
	@echo "  REAL DATA (needs .env with the NYU presigned URLs):"
	@echo "  make data            download val set + checkpoint into DATA_DIR, verify SHA256"
	@echo "  make extract         unpack the tarball (~2 h; xz here is single-threaded)"
	@echo "  make tier1           the real scored run -- these verdicts count"
	@echo "                       DATA_DIR=$(DATA_DIR)"
	@echo
	@echo "  make shell           interactive shell in the image, cwd mounted"
	@echo "  make report          print the verdicts from the last run"
	@echo "  make clean           remove $(RESULTS)/, $(SYNTH)/, jobtest/, tarballs"
	@echo "  make distclean       clean + drop the build stamp and the local image"
	@echo
	@echo "  make push IMAGE=<user>/fastmri-verify:2026-09      publish for CHTC"
	@echo
	@echo "         PARALLEL=64 CHUNK_MB=16    connections for the tarball download"
	@echo
	@echo "         SHM=8g                    /dev/shm for DataLoader workers"
	@echo "         WANDB=0                   disable W&B logging for tier1"
	@echo
	@echo "knobs: GPU=1  PYTEST=0  VOLUMES=  SLICES=  ACCEL='4 8'  CF='0.08 0.04'  EXTRA='...'"

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

push:
	@case "$(IMAGE)" in */*:*) ;; *) \
	  echo "IMAGE must be <dockerhub_user>/<name>:<dated-tag>, got '$(IMAGE)'" >&2; exit 1;; esac
	@case "$(IMAGE)" in *:latest) \
	  echo "refusing :latest -- a rebuild would silently change what past results were produced with" >&2; \
	  exit 1;; esac
	$(DOCKER) push $(IMAGE)

# --- fake data --------------------------------------------------------------
.PHONY: synthetic
synthetic: $(SYNTH)/multicoil_val

$(SYNTH)/multicoil_val: make_synthetic_val.py | build
	$(RUN) python make_synthetic_val.py --out $(SYNTH)/multicoil_val \
	    --volumes $(VOLUMES) --slices $(SLICES)
	@touch $@

knee_multicoil_val_synthetic.tar: $(SYNTH)/multicoil_val
	tar -cf $@ -C $(SYNTH) multicoil_val
	@ls -la $@

# --- tier 0 -----------------------------------------------------------------
# Real verdicts: a failure here means the image is wrong and nothing produced
# from it counts, so this one is not allowed to exit non-zero.
.PHONY: tier0
tier0: build
	@mkdir -p $(RESULTS)
	$(RUN) python verify_varnet.py tier0 \
	    --output_dir $(RESULTS) --fastmri_repo /opt/fastMRI $(PYTESTFLAG) --no_wandb $(EXTRA)

# --- tier 1 smoke -----------------------------------------------------------
# A random model on fake phantoms: the reference comparison and
# beats_zero_filled are EXPECTED to say FAIL, and the harness exits 1 for that.
# What is being tested is that file parsing, masking, model I/O, the metric
# code, the CSV/JSON writers and the verdict logic all run end to end, so the
# pass condition here is "a report was produced", not "the verdicts are green".
.PHONY: tier1-smoke
tier1-smoke: build synthetic
	@mkdir -p $(RESULTS)
	@rm -f $(RESULTS)/tier1_report.json
	-$(RUN) python verify_varnet.py tier1 \
	    --data_path $(SYNTH)/multicoil_val --output_dir $(RESULTS) \
	    --fastmri_repo /opt/fastMRI --random_init $(SMOKE_ARCH) \
	    --accelerations $(ACCEL) --center_fractions $(CF) \
	    --no_wandb $(CPUFLAG) --num_workers 0 $(EXTRA)
	@test -s $(RESULTS)/tier1_report.json || { \
	  echo; echo "FAILED: the harness crashed before writing $(RESULTS)/tier1_report.json"; exit 1; }
	@echo
	@echo "harness ran end to end. Reference-comparison and beats_zero_filled FAILs"
	@echo "are expected here: random weights, fake data."

# --- job executable ---------------------------------------------------------
# run_verify.sh is what HTCondor actually executes, and it does things the
# harness does not: tarball extraction, data discovery, the results/ contract
# for transfer_output_files. Running it locally against a synthetic tarball is
# the last thing that can be checked without a cluster slot.
# CRLF endings are fatal for anything that runs on Linux: bash reads the \r as
# part of each command ("$'\r': command not found", then a syntax error), and
# HTCondor would carry it into .sub macro values. .gitattributes keeps them LF
# on checkout; this catches a Windows editor putting them back.
.PHONY: check-eol
check-eol:
	@bad=""; \
	for f in run_verify.sh submit.sh prepare_staging.sh verify.sub verify_tier0.sub; do \
	  [ -f "$$f" ] || continue; \
	  if LC_ALL=C grep -qU "$$(printf '\r')" "$$f" 2>/dev/null; then bad="$$bad $$f"; fi; \
	done; \
	if [ -n "$$bad" ]; then \
	  echo "CRLF line endings found in:$$bad" >&2; \
	  echo "These execute on Linux and bash will choke on the carriage returns." >&2; \
	  echo "Fix:  sed -i 's/[carriage-return]\$$//'$$bad   (or: dos2unix$$bad)" >&2; \
	  exit 1; \
	fi

.PHONY: job-smoke
job-smoke: build check-eol knee_multicoil_val_synthetic.tar
	rm -rf jobtest && mkdir -p jobtest
	cp verify_varnet.py run_verify.sh knee_multicoil_val_synthetic.tar jobtest/
	-$(DOCKER) run --rm --platform $(PLATFORM) $(SHMFLAG) $(GPUFLAG) $(USERFLAG) $(ENVFLAGS) -e KEEP_TARBALL=1 \
	    -v "$(HOSTPWD)":/work -w /work/jobtest $(IMAGE) \
	    bash run_verify.sh tier1 --random_init $(SMOKE_ARCH) \
	        --accelerations $(ACCEL) --center_fractions $(CF) \
	        --no_wandb $(CPUFLAG) --num_workers 0 --volume_limit $(VOLUMES) $(EXTRA)
	@test -s jobtest/results/tier1_report.json || { \
	  echo; echo "FAILED: run_verify.sh produced no jobtest/results/tier1_report.json"; exit 1; }
	@echo
	@echo "run_verify.sh works: extraction, data discovery, results/ contract."
	@echo "The same tarball can smoke-test CHTC -- README.md section 2."

# --- real data --------------------------------------------------------------
# DATA_DIR deliberately defaults OUTSIDE the repo. This checkout sits under
# OneDrive, which would try to sync ~200 GiB of k-space to the cloud, and the
# .h5 files would also land next to tracked files. $(HOME) on Git Bash is
# C:/Users/<you>, which OneDrive does not sync.
# $(subst) because $(HOME) is a backslash path on Windows make (C:\Users\you),
# and backslashes get eaten as escapes once the value reaches sh.
DATA_DIR ?= $(subst \,/,$(HOME))/fastmri-data
VOLUME_LIMIT ?= 0
# Connections for the tarball download. S3 throttles per connection at about
# 0.2 MB/s regardless of how fat the local link is, so this is the only knob
# that matters: measured here, 1 conn = 0.20 MB/s (134 h), 16 = 3.3 (8 h),
# 64 = 13 (2 h), 128 = 22 (1.2 h), 192 = 28 (0.9 h) with no errors at any of
# them. This is NOT an S3 property: the ISP here (ResTech AS46262) shapes every
# TCP flow to ~2 Mbit/s for off-net traffic -- a non-AWS host in Newark meters
# out at the same 0.24 MB/s per flow, while CDN content cached on-net pulls at
# 47 MB/s on a single connection. Aggregate capacity is fine; per-flow is not,
# which is exactly the condition parallel range requests exist to defeat.
# 64 is a deliberate step back from the measured optimum: 128 workers cost
# ~13 MB/s more but held enough buffers to destabilise the desktop.
# PARALLEL=1 restores a plain single-stream curl (and on a normal network,
# where there is no per-flow shaping, 1 is the right answer).
PARALLEL ?= 64
# A chunk only reports progress when it finishes, and at ~0.24 MB/s per flow a
# 64 MB chunk takes 4.5 min -- long enough that the run looks hung. 16 MB
# chunks land every ~70s, which also means an interruption costs less.
CHUNK_MB ?= 16

# Resolve DATA_DIR to a host path docker accepts, creating it first.
# (Done inside the recipe, not at parse time, because the directory may not
# exist yet and `pwd -W` needs it to.)
define in_data_container
D=$$(mkdir -p "$(DATA_DIR)" && cd "$(DATA_DIR)" && { pwd -W 2>/dev/null || pwd; }); \
echo "[make] DATA_DIR -> $$D"; \
$(DOCKER) run --rm --platform $(PLATFORM) $(SHMFLAG) $(GPUFLAG) $(USERFLAG) $(ENVFLAGS)
endef

# Download the real val set + released checkpoint, and verify the SHA256.
# Reads the NYU presigned URLs from .env. Idempotent: prepare_staging.sh skips
# files whose size already matches the server and resumes partial ones, so an
# interrupted 94 GiB download can just be re-run.
.PHONY: data
data: build check-eol
	@test -f .env || { echo "no .env -- cp .env.example .env and paste the NYU URLs" >&2; exit 1; }
	@set -a; . ./.env; set +a; \
	test -n "$$FASTMRI_VAL_URL" -a -n "$$FASTMRI_SHA_URL" || { \
	  echo "FASTMRI_VAL_URL / FASTMRI_SHA_URL are empty in .env" >&2; exit 1; }
	@set -a; . ./.env; set +a; \
	$(in_data_container) \
	    -e FASTMRI_VAL_URL -e FASTMRI_SHA_URL -e STAGING_DIR=/data -e NETID \
	    -e PARALLEL=$(PARALLEL) -e CHUNK_MB=$(CHUNK_MB) \
	    -v "$(HOSTPWD)":/work -v "$$D":/data -w /work $(IMAGE) \
	    bash prepare_staging.sh

# Extract the tarball in place. Kept separate from `data` so a failed download
# is not re-extracted, and so this can be re-run without re-downloading.
.PHONY: extract
extract: build
	@$(in_data_container) -v "$$D":/data $(IMAGE) bash -c '\
	  if [ -d /data/multicoil_val ]; then echo "already extracted"; exit 0; fi; \
	  cd /data && echo "extracting, ~2 h -- no progress output until it finishes..." && \
	  xz -dc -T0 knee_multicoil_val.tar.xz | tar -x && \
	  ls -1 /data/multicoil_val/*.h5 | wc -l | xargs echo "volumes:"'

# The real scored run. Unlike tier1-smoke these verdicts MEAN something, so
# this target is not allowed to exit non-zero.
.PHONY: tier1
tier1: build
	@mkdir -p $(RESULTS)
	@if [ -f .env ]; then set -a; . ./.env; set +a; fi; \
	$(in_data_container) $(WANDBENV) \
	    -v "$(HOSTPWD)":/work -v "$$D":/data -w /work $(IMAGE) \
	    python verify_varnet.py tier1 \
	        --data_path /data/multicoil_val \
	        --state_dict /data/knee_leaderboard_state_dict.pt \
	        --output_dir $(RESULTS) --fastmri_repo /opt/fastMRI \
	        --accelerations $(ACCEL) --center_fractions $(CF) \
	        --mask_type $(MASK_TYPE) --volume_limit $(VOLUME_LIMIT) \
	        $(WANDBFLAG) $(CPUFLAG) $(EXTRA)

# --- the whole thing --------------------------------------------------------
# $(MAKE) is quoted because on Windows it expands to a path containing spaces
# and parentheses ("C:/Program Files (x86)/GnuWin32/bin/make"), which is a
# shell syntax error unquoted. The literal $(MAKE) still appears in the line,
# so make's recursive-invocation handling (MAKEFLAGS, -n) is unaffected.
.PHONY: local-run
local-run:
	@echo "=== 0/4  clean slate ============================================="
	"$(MAKE)" clean
	@echo "=== 1/4  image ==================================================="
	"$(MAKE)" build
	@echo "=== 2/4  tier0 ==================================================="
	"$(MAKE)" tier0
	@echo "=== 3/4  tier1 on synthetic data ================================="
	"$(MAKE)" tier1-smoke
	@echo "=== 4/4  job executable =========================================="
	"$(MAKE)" job-smoke
	@echo
	@echo "================================================================="
	@echo "local run complete. Reports:"
	@echo "  $(RESULTS)/tier0_report.json   real verdicts, all should be pass"
	@echo "  $(RESULTS)/tier1_report.json   smoke test, FAILs expected"
	@echo "  $(RESULTS)/per_volume_R*.csv, $(RESULTS)/results.csv"
	@echo "Next: push a dated image and stage the real data (README.md sections 1, 3)."
	@"$(MAKE)" --no-print-directory report

.PHONY: report
report:
	@for f in $(RESULTS)/tier0_report.json $(RESULTS)/tier1_report.json jobtest/results/tier1_report.json; do \
	  [ -s "$$f" ] || continue; \
	  printf '%-38s overall=%s\n' "$$f" \
	    "$$(sed -n 's/.*"overall": *"\([a-z]*\)".*/\1/p' "$$f" | head -1)"; \
	done

# --- misc -------------------------------------------------------------------
.PHONY: shell
shell: build
	$(DOCKER) run --rm $(SHMFLAG) -it $(GPUFLAG) $(USERFLAG) $(ENVFLAGS) \
	    -v "$(HOSTPWD)":/work -w /work $(IMAGE) bash

.PHONY: clean distclean
clean:
	rm -rf $(RESULTS) $(SYNTH) jobtest knee_multicoil_val_synthetic.tar

distclean: clean
	rm -rf .make logs runs
	-$(DOCKER) rmi $(IMAGE)
