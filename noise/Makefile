# noise/: measure the measurement noise already in the brain data (README.md)
.PHONY: test job-smoke submit submit-test summary

SYNTH := ../verification/brain_multicoil_val_batch_synthetic.tar.xz
IMAGE ?= genjigod/fastmri-verify:2026-09-18
SMOKE := .smoke

# estimator tests, numpy only
test:
	python3 test_measure_noise.py

# run_noise.sh exactly as HTCondor runs it, in the job image, on synthetic data
job-smoke:
	rm -rf $(SMOKE) && mkdir -p $(SMOKE)
	cp run_noise.sh measure_noise.py $(SYNTH) $(SMOKE)/
	docker run --rm --platform linux/amd64 -v $(CURDIR)/$(SMOKE):/w -w /w $(IMAGE) bash run_noise.sh val
	python3 measure_noise.py --summarize $(SMOKE)/output

# on the access point
submit:
	mkdir -p logs runs && condor_submit noise.sub
submit-test:
	mkdir -p logs runs && condor_submit noise.sub volume_limit=3

summary:
	python3 measure_noise.py --summarize runs/*/*/
