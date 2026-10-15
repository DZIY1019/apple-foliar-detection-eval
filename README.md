# Split-Integrity Auditing & Paired Bootstrap Evaluation
### Code release for "Semi-supervised detection of apple foliar diseases under extreme annotation scarcity: an integrity-audited protocol and seed–architecture decoupling for trusted orchard monitoring"

This repository contains the evaluation and auditing code that backs the two
methodological contributions of the paper:

1. **`split_integrity/`** — a split-integrity auditing tool that freezes the
   train/val/test partition and verifies, with both exact filename matching and
   pixel-level dHash near-duplicate detection, that the three splits are mutually
   exclusive, that the validation set contains no pseudo-labels, and that the
   sealed test set does not leak into any training or pseudo-labelled data.
2. **`evaluation/`** — a paired per-image bootstrap tool that produces the
   confidence intervals for every detection metric reported in the paper
   (mAP50, mAP50_clean, per-class AP50), with a reconciliation gate that
   re-derives full-sample per-class AP50 from the captured per-image inputs and
   requires exact agreement with the detector framework's official metrics
   before any interval is released.

The YOLO-family detector weights, training recipes, and the dataset itself are
described in the manuscript; the annotated dataset is publicly available at
[Zenodo](https://doi.org/10.5281/zenodo.23063925).

---

## Provenance and fidelity

Both tools are extracted from the research harness that produced every number
in the paper. The extraction is *logic-invariant*: comments were translated to
English and the machine-specific orchestration (training queue, per-arm
registry, GPU locking) was removed, but the statistical and auditing logic —
including the reconciliation gate, the paired resampling scheme, and the
per-image order-compatibility check — is byte-faithful to the research version.
Two design choices are worth knowing before you read the code:

* **AP is never re-implemented.** Per-image statistics are captured from the
  detector framework's own validation pass, and AP50 is always computed by the
  framework's `ap_per_class`. The tool only assembles per-image inputs.
* **The tool refuses to output numbers it cannot reconcile.** If the
  full-sample recomputation disagrees with the framework's official metrics
  beyond 1e-6, or if per-class support counts cannot be resolved, the tool
  exits with an error instead of emitting an interval.

---

## Repository layout

```
split_integrity/make_split_manifest.py   # freeze & audit train/val/test disjointness
evaluation/paired_bootstrap_ci.py        # per-image paired bootstrap CIs with reconciliation gate
requirements.txt
LICENSE (MIT)
```

## Environment

* Python 3.10
* ultralytics 8.4.152 (any >=8.3 should work; the tool fails loudly rather than
  silently on framework API changes)
* PyTorch (a GPU is required only for the validation pass in
  `paired_bootstrap_ci.py`; the self-test and the split audit are CPU-only)

```bash
pip install -r requirements.txt
```

## Usage

### 1. Audit the splits (CPU-only)

```bash
python split_integrity/make_split_manifest.py \
    --dataset  datasets/apple_pest_ssl \
    --val-new  datasets/_val_new/images \
    --test-dir datasets/apple_pest_ssl_test \
    --threshold 2
```

Writes `split_manifest.json` with per-split counts, the exact-name overlap
check, and all dHash near-duplicate hits between the test set and the
train/val reference pool. Exits non-zero if any assertion fails.

### 2. Paired bootstrap intervals (one GPU pass per arm)

```bash
python evaluation/paired_bootstrap_ci.py \
    --data   datasets/apple_pest_ssl/data.yaml \
    --ckpt-a runs/supervised/weights/best.pt \
    --ckpt-b runs/semi_supervised/weights/best.pt \
    --label-a supervised_371 --label-b semi_supervised_1297 \
    --n-boot 2000 --device 0
```

Prints and writes a markdown table of Δ (B−A) with per-image paired 95%
bootstrap intervals for `mAP50_clean` (primary; excludes the class given by
`--drop-class`), `mAP50`, and per-class AP50, plus a JSON sidecar with the full
result. Both checkpoints are evaluated on the same split in the same order;
per-image class-set equality is asserted before pairing.

### 3. Self-test (CPU-only, synthetic data)

```bash
python evaluation/paired_bootstrap_ci.py --selftest
```

## Citation

If you use this code, please cite the manuscript above. The annotated dataset:

> Deng, Z., Suo, X. (2026). Apple foliar disease detection dataset. Zenodo.
> https://doi.org/10.5281/zenodo.23063925

## License

MIT — see [LICENSE](LICENSE).
