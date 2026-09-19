# Makefile: driver for DPI VarNet training on CHTC.
#
# Two kinds of targets, run in two places:
#   laptop (needs Docker):        make test smoke job-smoke
#   ap2001.chtc.wisc.edu:         make submit resume status logs why
#
#   make help            every target
#
# The Docker image is ../training's, unchanged: the DPI code is transferred
# with the job, not baked in. Tests run in the verification image, which is
# the same environment plus pytest.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# --- knobs ------------------------------------------------------------------
# Comments go above the assignments, never after them: make keeps the
# trailing whitespace.
IMAGE      ?= fastmri-train:local
TEST_IMAGE ?= genjigod/fastmri-verify:2026-09-18
PLATFORM   ?= linux/amd64
# extra name=value macro overrides for condor_submit
ARGS       ?=
# W&B run name (spaces allowed); empty = "dpi mixed acceleration brain"
NAME       ?=
NAMEARG     = $(if $(NAME),name="$(NAME)")
# trained baseline checkpoint to warm start both parameter sets from
INIT       ?=
INITARG     = $(if $(INIT),init=$(INIT))
# checkpoint for `make resume`
CKPT       ?=
ENV_FILE   ?= ../.env
GPU        ?= $(if $(shell nvidia-smi -L 2>/dev/null),1,0)
SHM        ?= 8g
EXTRA      ?=

# Tiny architecture for the smoke targets: the point is the plumbing, not the
# model. BASE_ARCH is the same without the DPI-only flag, for the baseline
# checkpoint that warm-smoke starts from.
BASE_ARCH  ?= --num_cascades 2 --chans 4 --sens_chans 4 --pools 2 --sens_pools 2
SMOKE_ARCH ?= $(BASE_ARCH) --lambda_length 64
SMOKE_RUN  ?= --limit_train_batches 4 --limit_val_batches 2 --num_workers 0 --no_wandb

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
# WANDB_ALLOW_OFFLINE=1: the local targets have no key and log offline on
# purpose. On CHTC that variable is 0 unless OFFLINE=1 ./submit.sh, and a job
# without a working W&B connection holds itself.
ENVFLAGS := -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache -e MPLCONFIGDIR=/tmp/.cache/mpl \
            -e PYTHONUNBUFFERED=1 -e WANDB_MODE=offline -e WANDB_ALLOW_OFFLINE=1
RUN      := $(DOCKER) run --rm --platform $(PLATFORM) --shm-size=$(SHM) $(GPUFLAG) $(USERFLAG) $(ENVFLAGS) \
            -v "$(REPO)":/work -w /work/dpi $(IMAGE)
TESTRUN  := $(DOCKER) run --rm --platform $(PLATFORM) --shm-size=$(SHM) $(USERFLAG) $(ENVFLAGS) \
            -v "$(REPO)":/work -w /work/dpi $(TEST_IMAGE)

SYNTH := ../training/synthetic/knee

# ============================================================================

.PHONY: help
help:
	@echo "DPI VarNet (IMAGE=$(IMAGE))"
	@echo
	@echo "  LAPTOP (Docker):"
	@echo "  make test            unit tests: the DPI mechanics (21 checks)"
	@echo "  make smoke           train_dpi.py on synthetic data, then a rerun that MUST resume"
	@echo "  make job-smoke       run_train.sh with TRAIN_DRIVER=train_dpi.py, as HTCondor runs it"
	@echo "  make warm-smoke      warm start from a baseline checkpoint, then train"
	@echo "  make local-run       test + smoke + job-smoke + warm-smoke"
	@echo
	@echo "  ACCESS POINT (ap2001.chtc.wisc.edu, key in ../.env):"
	@echo "  make submit          brain, 2 4 6 8, from scratch; W&B 'dpi mixed acceleration brain'"
	@echo "  make submit INIT=../training/runs/brain/model2/<C>/checkpoints/last.ckpt"
	@echo "  make submit NAME='dpi mixed acceleration brain v2' ARGS='max_epochs=2'"
	@echo "  make resume CKPT=runs/brain/dpi/<Cluster>/checkpoints/last.ckpt"
	@echo "  make status          condor_q for your jobs"
	@echo "  make logs            tail the newest job .out"
	@echo "  make why JOB=<Cluster>   hold reason + log tails for a held job"
	@echo
	@echo "  make clean           remove smoke artefacts"
	@echo
	@echo "knobs: IMAGE= TEST_IMAGE= NAME= INIT= ARGS= CKPT= GPU= SHM= EXTRA='...'"

# --- line endings -----------------------------------------------------------
# These files execute on Linux; a stray CRLF makes bash fail on "\r" and turns
# a make variable into "linux/amd64\r".
.PHONY: check-eol
check-eol:
	@bad=""; \
	for f in Makefile submit.sh train.sub train_dpi.py dpi_varnet.py dpi_module.py dpi_transforms.py test_dpi.py; do \
	  if LC_ALL=C grep -qU "$$(printf '\r')" "$$f" 2>/dev/null; then bad="$$bad $$f"; fi; \
	done; \
	if [ -n "$$bad" ]; then \
	  echo "CRLF line endings found in:$$bad  (fix: dos2unix, or sed -i 's/\r$$//')" >&2; exit 1; \
	fi

# --- tests ------------------------------------------------------------------
# The evidence that the DPI mechanics are right: equivalence to the baseline
# at init, lambda monotone from 0 to 1, state-dict compatibility, parameter
# counts, gradient flow into both sets, the recorded acceleration.
.PHONY: test
test: check-eol
	$(TESTRUN) python -m pytest test_dpi.py -q

# --- synthetic data ---------------------------------------------------------
$(SYNTH)/multicoil_val:
	$(MAKE) -C ../training synthetic IMAGE=$(IMAGE)

.PHONY: synthetic
synthetic: $(SYNTH)/multicoil_val

# --- smoke: the driver ------------------------------------------------------
# Epoch 1 from scratch, then epoch 2 must start from output/checkpoints/last.ckpt.
.PHONY: smoke
smoke: check-eol synthetic
	rm -rf output
	$(RUN) python train_dpi.py --data_path $(SYNTH) --default_root_dir output \
	    --run_name dpi-smoke --accelerations 2 4 6 8 --center_fractions 0.16 0.08 0.0533 0.04 \
	    --gpus $(GPUS) --max_epochs 1 $(SMOKE_ARCH) $(SMOKE_RUN) $(EXTRA)
	@test -s output/checkpoints/last.ckpt || { echo "FAILED: no output/checkpoints/last.ckpt after epoch 1" >&2; exit 1; }
	@echo "--- epoch 1 done, checkpoint written. Re-running with --max_epochs 2: must resume ---"
	$(RUN) python train_dpi.py --data_path $(SYNTH) --default_root_dir output \
	    --run_name dpi-smoke --accelerations 2 4 6 8 --center_fractions 0.16 0.08 0.0533 0.04 \
	    --gpus $(GPUS) --max_epochs 2 $(SMOKE_ARCH) $(SMOKE_RUN) $(EXTRA) 2>&1 | tee output/resume.log
	@grep -q "resume from: output/checkpoints/last.ckpt" output/resume.log || { \
	  echo "FAILED: second run did not resume from last.ckpt" >&2; exit 1; }
	@echo
	@echo "smoke passed: DPI trains, last.ckpt is written, and a rerun resumes from it."

# --- smoke: warm start ------------------------------------------------------
# --init_from_baseline is the paper's warm start: both parameter sets begin at
# a trained baseline. This builds a baseline checkpoint with ../training's own
# driver, then starts DPI from it.
.PHONY: warm-smoke
warm-smoke: check-eol synthetic
	rm -rf warmtest && mkdir -p warmtest
	$(RUN) python ../training/train_wandb.py --data_path $(SYNTH) --default_root_dir warmtest/baseline \
	    --run_name warm-baseline --accelerations 2 4 6 8 --center_fractions 0.16 0.08 0.0533 0.04 \
	    --gpus $(GPUS) --max_epochs 1 $(BASE_ARCH) $(SMOKE_RUN) $(EXTRA)
	@test -s warmtest/baseline/checkpoints/last.ckpt || { echo "FAILED: no baseline checkpoint" >&2; exit 1; }
	$(RUN) python train_dpi.py --data_path $(SYNTH) --default_root_dir warmtest/dpi \
	    --run_name dpi-warm --accelerations 2 4 6 8 --center_fractions 0.16 0.08 0.0533 0.04 \
	    --gpus $(GPUS) --max_epochs 1 $(SMOKE_ARCH) $(SMOKE_RUN) $(EXTRA) \
	    --init_from_baseline warmtest/baseline/checkpoints/last.ckpt 2>&1 | tee warmtest/warm.log
	@grep -q "warm start from last.ckpt" warmtest/warm.log || { \
	  echo "FAILED: --init_from_baseline did not load the baseline" >&2; exit 1; }
	@echo
	@echo "warm-smoke passed: both parameter sets start from a trained baseline."

# --- smoke: the job executable ----------------------------------------------
# ../training/run_train.sh with TRAIN_DRIVER=train_dpi.py, exactly as train.sub
# runs it: tarball extraction, split discovery, the W&B preflight, the output/
# contract. Run twice in the same scratch dir; the second run is "after an
# eviction" and must resume.
.PHONY: job-smoke
job-smoke: check-eol synthetic
	$(MAKE) -C ../training synthetic_multicoil_train.tar IMAGE=$(IMAGE)
	rm -rf jobtest && mkdir -p jobtest
	cp ../training/train_wandb.py ../training/run_train.sh train_dpi.py dpi_varnet.py \
	   dpi_module.py dpi_transforms.py ../training/synthetic_multicoil_*.tar jobtest/
	$(DOCKER) run --rm --platform $(PLATFORM) --shm-size=$(SHM) $(GPUFLAG) $(USERFLAG) $(ENVFLAGS) \
	    -e KEEP_TARBALL=1 -e TRAIN_DRIVER=train_dpi.py \
	    -v "$(REPO)":/work -w /work/dpi/jobtest $(IMAGE) \
	    bash run_train.sh --run_name dpi-job-smoke --accelerations 2 4 6 8 \
	        --center_fractions 0.16 0.08 0.0533 0.04 \
	        --gpus $(GPUS) --max_epochs 1 $(SMOKE_ARCH) $(SMOKE_RUN) $(EXTRA)
	@test -s jobtest/output/checkpoints/last.ckpt || { echo "FAILED: run_train.sh produced no checkpoint" >&2; exit 1; }
	@echo "--- first job done. Second run in the same sandbox (= after an eviction): must resume ---"
	rm -rf jobtest/data
	$(DOCKER) run --rm --platform $(PLATFORM) --shm-size=$(SHM) $(GPUFLAG) $(USERFLAG) $(ENVFLAGS) \
	    -e KEEP_TARBALL=1 -e TRAIN_DRIVER=train_dpi.py \
	    -v "$(REPO)":/work -w /work/dpi/jobtest $(IMAGE) \
	    bash run_train.sh --run_name dpi-job-smoke --accelerations 2 4 6 8 \
	        --center_fractions 0.16 0.08 0.0533 0.04 \
	        --gpus $(GPUS) --max_epochs 2 $(SMOKE_ARCH) $(SMOKE_RUN) $(EXTRA) 2>&1 | tee jobtest/resume.log
	@grep -q "driver=train_dpi.py" jobtest/resume.log || { \
	  echo "FAILED: run_train.sh did not run the DPI driver" >&2; exit 1; }
	@grep -q "resume from: output/checkpoints/last.ckpt" jobtest/resume.log || { \
	  echo "FAILED: evicted-job rerun did not resume from last.ckpt" >&2; exit 1; }
	@echo
	@echo "job-smoke passed: the DPI driver runs under run_train.sh, and resume works."

.PHONY: local-run
local-run: test smoke job-smoke warm-smoke
	@echo
	@echo "local-run passed: mechanics, training, resume, warm start, and the job path."

# --- CHTC access point ------------------------------------------------------
.PHONY: submit resume status logs why
submit:
	./submit.sh $(NAMEARG) $(INITARG) $(ARGS)

resume:
	@test -n "$(CKPT)" || { echo "usage: make resume CKPT=runs/brain/dpi/<Cluster>/checkpoints/last.ckpt [NAME='...']" >&2; exit 1; }
	./submit.sh resume=$(CKPT) $(NAMEARG) $(ARGS)

status:
	condor_q -nobatch

logs:
	@f=$$(ls -t logs/*.out 2>/dev/null | head -1); \
	test -n "$$f" || { echo "no logs/*.out yet" >&2; exit 1; }; \
	echo "== $$f"; tail -f "$$f"

why:
	@test -n "$(JOB)" || { echo "usage: make why JOB=<Cluster>[.<Proc>]" >&2; exit 1; }
	@echo "== hold"; condor_q $(JOB) -af:t HoldReasonCode HoldReasonSubCode NumJobStarts NumSystemHolds \
	    LastRemoteHost RemoteWallClockTime MemoryUsage RequestMemory DiskUsage RequestDisk ExitCode ExitBySignal ExitSignal
	@condor_q $(JOB) -af HoldReason
	@c=$${JOB%%.*}; for f in logs/*_$$c.log logs/*_$$c_*.out logs/*_$$c_*.err; do \
	    [ -f "$$f" ] || continue; echo "== $$f"; tail -n 60 "$$f"; done

# --- misc -------------------------------------------------------------------
.PHONY: clean distclean
clean:
	rm -rf output jobtest warmtest baseline_init.ckpt

distclean: clean
	rm -rf logs runs
