# Makefile: driver for VarNet evaluation on CHTC. The same targets as
# ../training/Makefile, run in the same two places:
#   laptop (needs Docker):        make build push tier0 smoke job-smoke
#   ap2001.chtc.wisc.edu:         make submit-tier0 submit-model1 submit-model2 status logs
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
# IMAGE: use a dated tag (<user>/fastmri-verify:2026-09) for anything you push.
IMAGE    ?= fastmri-verify:local
PLATFORM ?= linux/amd64
# model preset for `make submit` and the local `make tier1`: model1 | model2
MODEL    ?= model1
# extra name=value macro overrides for condor_submit, e.g.
#   make submit-model1 ARGS='volume_limit=20 extra_args="--num_workers 0"'
ARGS     ?=
# synthetic volumes for the smoke tests
VOLUMES  ?= 3
SLICES   ?= 4
# GPU: 1 uses the host GPU, 0 forces CPU. Auto-detected from nvidia-smi.
GPU      ?= $(if $(shell nvidia-smi -L 2>/dev/null),1,0)
SHM      ?= 8g
# extra args appended to verify_varnet.py in the smoke and tier1 targets
EXTRA    ?=
# PYTEST=1: tier0 also runs the fastMRI repo test suite
PYTEST   ?= 1

# The model presets: the same lists submit.sh applies on the access point,
# paired elementwise as in fastmri.data.subsample.MaskFunc. Override either
# list on the command line (ACCEL='4 8' CF='0.08 0.04').
ifeq ($(MODEL),model2)
  ACCEL ?= 2 4 6 8
  CF    ?= 0.16 0.08 0.0533 0.04
else
  ACCEL ?= 4
  CF    ?= 0.08
endif
# equispaced_fraction is what training uses. random is the fastMRI knee
# convention and the only family the reference values in verify_varnet.py
# are valid for (README "Mask family notes").
MASK_TYPE ?= equispaced_fraction

# Tiny architecture for the smoke tests: the point is exercising the data
# path, the metric code, the report writers and the job executable, not the
# model. --random_init skips checkpoint loading, so the numbers are noise.
SMOKE_ARCH ?= --num_cascades 2 --chans 4 --sens_chans 4 --pools 2 --sens_pools 2
SMOKE_RUN  ?= --random_init --num_workers 0

# --- host/docker plumbing ---------------------------------------------------
# The repo root is mounted (not just this directory), the same as training/,
# so the container sees ../training too and paths match between the two.
#
# Windows: Git Bash mangles both the host path and the container-side path.
# `pwd -W` gives the C:/... form Docker Desktop wants (plain $(CURDIR) yields
# /c/... , which Docker rejects as a bind mount source) and MSYS_NO_PATHCONV=1
# stops Git Bash rewriting `-w /work/verification` into a C:/Program Files/...
# path. --user is dropped there too: Docker Desktop already maps the bind
# mount as root, and a Git Bash uid (197609) does not exist inside the image.
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
# W&B mode is not fixed here (training's ENVFLAGS pins offline) because the
# local `make tier1` is a run worth logging online; the smoke targets pin it.
ENVFLAGS := -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache -e MPLCONFIGDIR=/tmp/.cache/mpl \
            -e PYTHONUNBUFFERED=1
RUN      := $(DOCKER) run --rm --platform $(PLATFORM) --shm-size=$(SHM) $(GPUFLAG) $(USERFLAG) $(ENVFLAGS) \
            -v "$(REPO)":/work -w /work/verification $(IMAGE)

STAMP := .make/image.$(subst :,_,$(subst /,_,$(IMAGE)))
SYNTH := synthetic/knee

# ============================================================================

.PHONY: help
help:
	@echo "VarNet evaluation on CHTC (IMAGE=$(IMAGE))"
	@echo
	@echo "  LAPTOP (Docker):"
	@echo "  make build           build the image for $(PLATFORM) (cached)"
	@echo "  make push IMAGE=<user>/fastmri-verify:2026-09   publish for CHTC"
	@echo "  make tier0           environment + metric invariants (+ fastMRI pytest); real verdicts"
	@echo "  make smoke           verify_varnet.py on synthetic data with a random model"
	@echo "  make job-smoke       run_verify.sh the way HTCondor will, with an offline W&B run"
	@echo "  make shell           interactive shell in the image, repo mounted at /work"
	@echo
	@echo "  ACCESS POINT (ap2001.chtc.wisc.edu, WANDB_API_KEY exported):"
	@echo "  make submit-tier0    environment checks on a GPU node, once per image tag"
	@echo "  make submit-model1   score the checkpoint at acceleration 4"
	@echo "  make submit-model2   score the checkpoint at accelerations 2 4 6 8, one pass per rate"
	@echo "  make submit MODEL=model2 ARGS='ckpt=runs/model2/<Cluster>/checkpoints/last.ckpt'"
	@echo "  make status          condor_q for your jobs"
	@echo "  make logs            tail the newest job .out"
	@echo
	@echo "  LAPTOP, REAL DATA (needs .env with the NYU presigned URLs):"
	@echo "  make data            download val set + checkpoint into DATA_DIR, verify SHA256"
	@echo "  make extract         unpack the tarball (~2 h)"
	@echo "  make tier1           the real scored run on DATA_DIR=$(DATA_DIR)"
	@echo
	@echo "  make report          print the verdicts from the last local run"
	@echo "  make clean           remove smoke artefacts;  make distclean also drops logs/ runs/ and the image"
	@echo
	@echo "knobs: IMAGE= PLATFORM= MODEL= ARGS= GPU= SHM= VOLUMES= SLICES= PYTEST=0 EXTRA='...'"
	@echo "       MODEL=model1|model2 picks ACCEL/CF; MASK_TYPE=$(MASK_TYPE); PARALLEL=$(PARALLEL) for the download"

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
	  echo "refusing :latest -- a rebuild would silently change what past results were produced with" >&2; \
	  exit 1;; esac
	$(DOCKER) push $(IMAGE)
	@echo
	@echo "now set in verify.sub and verify_tier0.sub:   image = docker://$(IMAGE)"

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
	for f in Makefile Dockerfile run_verify.sh submit.sh prepare_staging.sh verify.sub verify_tier0.sub verify_varnet.py make_synthetic_val.py; do \
	  if LC_ALL=C grep -qU "$$(printf '\r')" "$$f" 2>/dev/null; then bad="$$bad $$f"; fi; \
	done; \
	if [ -n "$$bad" ]; then \
	  echo "CRLF line endings found in:$$bad  (fix: dos2unix, or sed -i 's/\r$$//')" >&2; exit 1; \
	fi

# --- synthetic data ---------------------------------------------------------
# Phantoms with the real on-disk layout. Numbers scored on this are
# meaningless; it only exercises the pipeline.
.PHONY: synthetic
synthetic: $(SYNTH)/multicoil_val

$(SYNTH)/multicoil_val: make_synthetic_val.py | build
	$(RUN) python make_synthetic_val.py --out $(SYNTH)/multicoil_val \
	    --volumes $(VOLUMES) --slices $(SLICES) --seed 2
	@touch $@

knee_multicoil_val_synthetic.tar: $(SYNTH)/multicoil_val
	tar -cf $@ -C $(SYNTH) multicoil_val

# --- tier 0 -----------------------------------------------------------------
# Real verdicts: a failure here means the image is wrong and nothing produced
# from it counts, so this one is not allowed to exit non-zero.
.PHONY: tier0
tier0: build check-eol
	@mkdir -p output
	$(RUN) python verify_varnet.py tier0 \
	    --output_dir output --fastmri_repo /opt/fastMRI $(PYTESTFLAG) --no_wandb $(EXTRA)

# --- smoke: the harness -----------------------------------------------------
# A random model on fake phantoms with the model 2 rate list: the reference
# comparison and beats_zero_filled are EXPECTED to say FAIL, and the harness
# exits 1 for that. What is being tested is that file parsing, masking, model
# I/O, the metric code, the CSV/JSON writers and the verdict logic all run
# end to end, so the pass condition is "a report was produced".
.PHONY: smoke
smoke: build check-eol synthetic
	rm -rf output
	-$(RUN) python verify_varnet.py tier1 --data_path $(SYNTH)/multicoil_val --output_dir output \
	    --fastmri_repo /opt/fastMRI --run_name smoke \
	    --accelerations 2 4 6 8 --center_fractions 0.16 0.08 0.0533 0.04 --mask_type $(MASK_TYPE) \
	    $(SMOKE_ARCH) $(SMOKE_RUN) --no_wandb $(CPUFLAG) $(EXTRA)
	@test -s output/tier1_report.json || { echo "FAILED: the harness crashed before writing output/tier1_report.json" >&2; exit 1; }
	@echo
	@echo "smoke passed: the harness ran end to end. Reference-comparison and"
	@echo "beats_zero_filled FAILs are expected here: random weights, fake data."

# --- smoke: the job executable ----------------------------------------------
# run_verify.sh does what the harness does not: tarball extraction, split
# discovery, checkpoint discovery, the output/ contract. W&B runs offline
# here rather than being disabled so the run directory and its latest-run
# symlink exist, and the check at the end proves run_verify.sh removed the
# symlink -- HTCondor puts a job on hold if one is left in output/.
.PHONY: job-smoke
job-smoke: build check-eol knee_multicoil_val_synthetic.tar
	rm -rf jobtest && mkdir -p jobtest
	cp verify_varnet.py run_verify.sh knee_multicoil_val_synthetic.tar jobtest/
	-$(DOCKER) run --rm --platform $(PLATFORM) --shm-size=$(SHM) $(GPUFLAG) $(USERFLAG) $(ENVFLAGS) \
	    -e KEEP_TARBALL=1 -e WANDB_MODE=offline \
	    -v "$(REPO)":/work -w /work/verification/jobtest $(IMAGE) \
	    bash run_verify.sh tier1 --run_name job-smoke --accelerations 4 --center_fractions 0.08 \
	        --mask_type $(MASK_TYPE) --volume_limit $(VOLUMES) $(SMOKE_ARCH) $(SMOKE_RUN) $(CPUFLAG) $(EXTRA)
	@test -s jobtest/output/tier1_report.json || { echo "FAILED: run_verify.sh produced no output/tier1_report.json" >&2; exit 1; }
	@if [ -n "$$(find jobtest/output -type l)" ]; then \
	  echo "FAILED: symlinks left under jobtest/output (HTCondor would hold the job):" >&2; \
	  find jobtest/output -type l >&2; exit 1; fi
	@echo
	@echo "job-smoke passed: extraction, split discovery, checkpoint discovery, output/ contract, no symlinks."

# --- CHTC access point ------------------------------------------------------
.PHONY: submit submit-tier0 submit-model1 submit-model2 status logs
submit-tier0:
	./submit.sh tier0 $(ARGS)

submit-model1:
	./submit.sh model1 $(ARGS)

submit-model2:
	./submit.sh model2 $(ARGS)

submit:
	./submit.sh $(MODEL) $(ARGS)

status:
	condor_q -nobatch

logs:
	@f=$$(ls -t logs/*.out 2>/dev/null | head -1); \
	test -n "$$f" || { echo "no logs/*.out yet" >&2; exit 1; }; \
	echo "== $$f"; tail -f "$$f"

# --- laptop, real data ------------------------------------------------------
# Evaluation is the one job that can also run on a laptop with a GPU, because
# the val split is the only thing it needs. Nothing here exists in training/.
#
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
# WANDB=1 (default) forwards the credentials from .env and lets the harness
# choose: online when WANDB_API_KEY arrives, offline otherwise, in which case
# it prints the `wandb sync` command. WANDB=0 turns logging off entirely.
WANDB ?= 1
ifeq ($(WANDB),1)
  WANDBFLAG :=
  WANDBENV  := -e WANDB_API_KEY -e WANDB_ENTITY -e WANDB_PROJECT -e WANDB_MODE
else
  WANDBFLAG := --no_wandb
  WANDBENV  := -e WANDB_MODE=offline
endif

# Resolve DATA_DIR to a host path docker accepts, creating it first.
# (Done inside the recipe, not at parse time, because the directory may not
# exist yet and `pwd -W` needs it to.)
define in_data_container
D=$$(mkdir -p "$(DATA_DIR)" && cd "$(DATA_DIR)" && { pwd -W 2>/dev/null || pwd; }); \
echo "[make] DATA_DIR -> $$D"; \
$(DOCKER) run --rm --platform $(PLATFORM) --shm-size=$(SHM) $(GPUFLAG) $(USERFLAG) $(ENVFLAGS)
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
	    -v "$(REPO)":/work -v "$$D":/data -w /work/verification $(IMAGE) \
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

# The real scored run, with the same preset lists the CHTC job uses
# (MODEL=model1|model2). Unlike smoke these verdicts MEAN something, so this
# target is not allowed to exit non-zero.
.PHONY: tier1
tier1: build check-eol
	@mkdir -p output
	@if [ -f .env ]; then set -a; . ./.env; set +a; fi; \
	$(in_data_container) $(WANDBENV) \
	    -v "$(REPO)":/work -v "$$D":/data -w /work/verification $(IMAGE) \
	    python verify_varnet.py tier1 \
	        --data_path /data/multicoil_val \
	        --state_dict /data/knee_leaderboard_state_dict.pt \
	        --output_dir output --fastmri_repo /opt/fastMRI --run_name verify-$(MODEL)-local \
	        --accelerations $(ACCEL) --center_fractions $(CF) \
	        --mask_type $(MASK_TYPE) --volume_limit $(VOLUME_LIMIT) \
	        $(WANDBFLAG) $(CPUFLAG) $(EXTRA)

.PHONY: report
report:
	@for f in output/tier0_report.json output/tier1_report.json jobtest/output/tier1_report.json; do \
	  [ -s "$$f" ] || continue; \
	  printf '%-38s overall=%s\n' "$$f" \
	    "$$(sed -n 's/.*"overall": *"\([a-z]*\)".*/\1/p' "$$f" | head -1)"; \
	done

# --- misc -------------------------------------------------------------------
.PHONY: shell
shell: build
	$(DOCKER) run --rm -it --platform $(PLATFORM) --shm-size=$(SHM) $(GPUFLAG) $(USERFLAG) $(ENVFLAGS) \
	    -v "$(REPO)":/work -w /work/verification $(IMAGE) bash

.PHONY: clean distclean
clean:
	rm -rf output results synthetic jobtest knee_multicoil_val_synthetic.tar

distclean: clean
	rm -rf .make logs runs
	-$(DOCKER) rmi $(IMAGE)
