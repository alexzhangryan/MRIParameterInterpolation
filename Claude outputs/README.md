# Claude outputs/ — archived first draft, superseded

**Do not run anything in this directory.** It is the first-draft snapshot of
the verification harness, committed in `939e1c9` and superseded by
`../verification/` in `8bc82f7`. It is kept only as a record of what the
first pass looked like.

| File here | Live version | Divergence |
|---|---|---|
| `verify_varnet.py` | `../verification/verify_varnet.py` | 759 → 831 lines |
| `verify.sub` | `../verification/verify.sub` | 70 → 83 lines |
| `Dockerfile` | `../verification/Dockerfile` | 70 → 89 lines |
| `README.md` | `../verification/README.md` | this file |

The live directory additionally has `Makefile`, `run_verify.sh`, `submit.sh`,
`prepare_staging.sh`, `make_synthetic_val.py`, `verify_tier0.sub`, and
`.env.example` — none of which exist here, so this snapshot cannot be built
or submitted on its own.

Things this copy's README still said that are now wrong, and which is why it
should not be used as a runbook:

- `/staging/<netid>/...` — personal staging is sharded by the netid's first
  letter, so it is `/staging/a/apryan3/`
- `prepare_staging.sh` run from the access point — it belongs on
  `transfer.chtc.wisc.edu`, and it takes `PARALLEL=`
- `request_disk` of 230 GB and "20 to 40 minutes" to extract — measured at
  ~320 GB and roughly 2 hours
- `docker build` with no `--platform linux/amd64` — that produces an arm64
  image on this laptop and dies in the conda layer
- no mention of the 100 GB `/staging` quota or the subset repack that the
  whole project now depends on (`../training/README.md` section 3b)

Read `../verification/README.md` instead. If this directory is not wanted in
git history going forward, `git rm -r "Claude outputs"` is safe: nothing
references it.
