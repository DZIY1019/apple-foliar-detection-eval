#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired per-image bootstrap confidence intervals for detection metrics.

This is the evaluation core behind every paired bootstrap interval reported in
the manuscript "Semi-supervised detection of apple foliar diseases under
extreme annotation scarcity" (submitted to Computers and Electronics in
Agriculture). It is a logic-faithful extraction of the research harness; the
machine-specific orchestration (training queue, per-arm registry, GPU locks)
was removed. The statistical logic below is unchanged:

* AP is always computed by the detector framework's own `ap_per_class`; this
  tool never re-implements AP. Per-image statistics are captured from the
  framework's validation pass by intercepting the moment before the framework
  clears its per-image buffers.
* A reconciliation gate re-computes full-sample per-class AP50 from the
  captured inputs and requires exact agreement (tolerance 1e-6) with the
  framework's official `box.ap50`. If reconciliation fails, the tool refuses
  to emit any interval.
* Paired resampling applies the *same* image sample to both arms. The reported
  point estimate is the full-sample difference, NOT the mean of resampled
  deltas: for a non-linear statistic like AP, the bootstrap mean does not
  equal the point estimate, and mixing the two makes a table inconsistent
  with its own intervals.
* Per-image order compatibility (equal class sets per image) is asserted
  before pairing; image count alone is not sufficient.

Two hardening rules are inherited from the research version and matter for
anyone reusing this code:

* Empty per-image arrays must not break concatenation: a resample that draws
  only zero-prediction images is legal, so `(0,)` and `(0, N_IOU)` shapes are
  normalised before stacking.
* Class support counts are translated from class *indices* (what the framework
  stores) to class *names* before any label substitution; a direct
  index/name mix-up silently renders every support count as 0.

Usage:
  python paired_bootstrap_ci.py --data data.yaml \
      --ckpt-a supervised/best.pt --ckpt-b semi_supervised/best.pt --device 0
  python paired_bootstrap_ci.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

N_IOU = 10          # length of the framework's IoU vector (0.50:0.05:0.95)
AP_TOL = 1e-6       # full-sample recomputation vs. official AP tolerance

# Paper's class set: the class named in --drop-class (default apple_leafroller)
# is excluded from the primary metric because all of its images derive from
# 36 rotations of four source photographs spanning the train/val split.
DEFAULT_CLASSES = ("apple_powdery_mildew", "apple_scab", "apple_leafroller")
DEFAULT_EXCLUDED = ("apple_leafroller",)


# ---------------------------------------------------------------------------
# Per-image input assembly and AP recomputation
# ---------------------------------------------------------------------------
def _stack(arrays, cols=None):
    """Concatenate per-image arrays; zero-length items must not break the stack.

    A zero-prediction image yields shape `(0, N_IOU)` for `tp` but `(0,)` for
    `conf`/`pred_cls`; a naive `np.concatenate` raises on the mismatch. Each
    array is reshaped to the target column count (or flattened) first.
    """
    out = []
    for a in arrays:
        a = np.asarray(a)
        if a.size == 0:
            a = np.zeros((0, cols), dtype=np.float32) if cols else np.zeros(0, dtype=np.float32)
        elif cols is not None:
            a = a.reshape(-1, cols)
        else:
            a = a.reshape(-1)
        out.append(a)
    if not out:
        return np.zeros((0, cols), dtype=np.float32) if cols else np.zeros(0, dtype=np.float32)
    return np.concatenate(out, 0)


def ap50_per_class(stats, idx):
    """Recompute per-class AP50 on the given image-index subset.

    `idx` may contain repeats (bootstrap resampling) or be `range(n)` (the
    full sample, used by the reconciliation gate). AP itself comes entirely
    from the framework's `ap_per_class`; the position of each return value is
    asserted rather than assumed, so a framework API change fails loudly
    instead of silently feeding the wrong array into the metric.
    """
    from ultralytics.utils.metrics import ap_per_class

    tp = _stack([stats["tp"][i] for i in idx], cols=N_IOU)
    conf = _stack([stats["conf"][i] for i in idx])
    pred_cls = _stack([stats["pred_cls"][i] for i in idx])
    target_cls = _stack([stats["target_cls"][i] for i in idx])
    # No predictions and no targets -> AP undefined; return explicitly empty.
    if tp.shape[0] == 0 and target_cls.shape[0] == 0:
        return {}
    # Positional access on purpose: in the tested framework version (8.4.152)
    # `ap_per_class` returns 12 items; `ap` is [5] and `unique_classes` is [6].
    # Unpacking by name/count breaks loudly on version changes - keep it that way.
    res = ap_per_class(tp, conf, pred_cls, target_cls)
    ap, ap_class = res[5], res[6]
    if getattr(ap, "ndim", 0) != 2 or ap.shape[1] != N_IOU:
        raise RuntimeError(f"ap_per_class returned unexpected ap shape {getattr(ap, 'shape', None)} "
                           f"(expected (*, {N_IOU})); framework version may have changed")
    return {int(c): float(ap[i, 0]) for i, c in enumerate(ap_class)}


def summary_metrics(per_class: dict, names: list, observed=DEFAULT_CLASSES, excluded=DEFAULT_EXCLUDED):
    """Fold `{class_index: AP50}` into the paper's metric set.

    Missing classes become None, never 0: a class absent from the AP result
    must not masquerade as a zero-AP class, which would silently drag down the
    primary metric.
    """
    def get(name):
        return per_class.get(names.index(name)) if name in names else None

    raw = {n: get(n) for n in observed}
    trusted = [raw[n] for n in observed if n not in excluded and raw[n] is not None]
    allv = [raw[n] for n in observed if raw[n] is not None]
    out = {
        # Primary metric: mean AP50 over classes not excluded by the leakage audit.
        "mAP50_clean": (sum(trusted) / len(trusted)) if trusted else None,
        # Reference metric: mean over all observed classes (leak-inflated; contrast only).
        "mAP50": (sum(allv) / len(allv)) if allv else None,
    }
    for n in observed:
        out[n] = raw[n]
    return out


def class_support(stats, names: list) -> dict[str, int]:
    """Per-class ground-truth instance counts from the framework's per-image data.

    `target_cls` stores class *indices*; translate with the framework's `names`
    before anyone substitutes these into a name-keyed template. A direct
    `Counter(target_cls)` keyed by int would silently render every n as 0.
    """
    import collections

    cnt = collections.Counter(int(c) for arr in stats["target_cls"] for c in arr)
    return {names[i]: v for i, v in cnt.items() if 0 <= i < len(names)}


def metric_labels(support: dict, observed, excluded):
    """Human-readable metric labels with support counts substituted in.

    The guard is not decoration: a missing support entry would silently render
    as 0 and be read as truth. Hard-fail instead of emitting numbers.
    """
    bad = [c for c in observed if not support.get(c)]
    if bad:
        raise SystemExit(f"[err] no support count for {bad} (available: {sorted(support)}); "
                         f"refusing to emit metrics rather than report n=0")
    labels = {
        "mAP50_clean": f"mAP50_clean (primary; excludes {', '.join(excluded)})",
        "mAP50": "mAP50 (all classes; reference only)",
    }
    for c in observed:
        labels[c] = f"{c} AP50 (n={support[c]})"
    return labels


def consistency_check(stats, box, names):
    """Reconciliation gate: recomputed per-class AP50 must match official `box.ap50`.

    Returns `(ok, detail)`. Without this gate, any mis-assembly of the per-image
    inputs (wrong image order, wrong columns, wrong subsample) would produce
    intervals that look perfectly plausible and are simply wrong.
    """
    n = len(stats["tp"])
    mine = ap50_per_class(stats, range(n))
    idx = [int(c) for c in box.ap_class_index]
    theirs = {c: float(box.ap50[idx.index(c)]) for c in idx}
    diffs = {}
    for c in theirs:
        diffs[c] = abs(mine[c] - theirs[c]) if c in mine else float("inf")
    ok = bool(diffs) and all(v <= AP_TOL for v in diffs.values())
    detail = ", ".join(
        f"{names[c] if c < len(names) else c}: tool {mine.get(c, float('nan')):.6f} "
        f"vs official {theirs[c]:.6f} (d={v:.2e})"
        for c, v in diffs.items()
    )
    return ok, detail


# ---------------------------------------------------------------------------
# Capturing per-image statistics from the framework's validation pass
# ---------------------------------------------------------------------------
def _stats_keeper_cls():
    """Build a `DetectionValidator` subclass that snapshots per-image statistics.

    Why a subclass: during single-device validation the framework clears its
    per-image buffers inside `DetectionValidator.get_stats()` via
    `self.metrics.clear_stats()`; after that call the per-image data is gone.
    Wrapping *the metrics instance's* `clear_stats` lets us keep a copy just
    before the clear, without touching the framework's AP computation.

    The hook point must be the `metrics` INSTANCE, not the validator class:
    overriding `DetectionValidator.clear_stats` is silently ineffective because
    the validator has no such method - the framework calls
    `self.metrics.clear_stats()`. The failure mode is nasty (validation runs
    fine, aggregate metrics match, `per_image_stats` stays None forever), so
    the hook is installed in `__init__` on the instance attribute, before any
    clear can happen and regardless of whether `init_metrics()` runs.
    """
    from ultralytics.models.yolo.detect import DetectionValidator

    class StatsKeeper(DetectionValidator):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)          # self.metrics exists from here on
            self.per_image_stats = None
            metrics = self.metrics
            original = metrics.clear_stats

            def clear_stats():
                # Snapshot before clearing - the order is the whole point.
                # init_metrics() also clears once, while stats are still empty;
                # the falsy guard skips that call.
                if self.per_image_stats is None and metrics.stats.get("tp"):
                    self.per_image_stats = {k: list(v) for k, v in metrics.stats.items()}
                original()

            metrics.clear_stats = clear_stats   # instance attribute shadows the method

    return StatsKeeper


def build_val_args(overrides: dict, data: str, device: str, imgsz: int, workers: int | None = None) -> dict:
    """Assemble the validation args dict (pure function; no ultralytics import).

    Merge order mirrors `Model.val()`: `{**overrides, "rect": True, ...}`.
    Getting this wrong silently evaluates with different parameters than the
    rest of the paper.

    When `workers` is None the key must NOT be added: writing
    `args["workers"] = None` unconditionally would override the value stored in
    the checkpoint's training args and silently change the evaluation setup.
    """
    args = {**overrides, "rect": True, "data": data, "device": device,
            "imgsz": imgsz, "verbose": False, "plots": False, "save_json": False,
            "mode": "val"}
    if workers is not None:
        args["workers"] = workers
    return args


def collect(ckpt: Path, data: str, device: str, imgsz: int = 640, workers: int | None = None):
    """Run one official validation pass; return (per-image stats, official box, names, n_images)."""
    from ultralytics import YOLO

    model = YOLO(str(ckpt))
    args = build_val_args(model.overrides, data, device, imgsz, workers)
    validator = _stats_keeper_cls()(args=args, _callbacks=model.callbacks)
    validator(model=model.model)
    if not validator.per_image_stats:
        raise RuntimeError("failed to capture per-image statistics "
                           "(framework version changed? check where clear_stats is called)")
    names = [validator.metrics.names[i] for i in sorted(validator.metrics.names)]
    return validator.per_image_stats, validator.metrics.box, names, len(validator.per_image_stats["tp"])


# ---------------------------------------------------------------------------
# Bootstrap resampling
# ---------------------------------------------------------------------------
def bootstrap(stats, names, observed, excluded, n_boot: int, seed: int = 42, alpha: float = 0.05):
    """Per-image (unpaired) bootstrap; returns {metric: (point, lo, hi)}.

    The point estimate uses the FULL sample, never a resample: it must equal
    the value in the paper's tables, otherwise the table and its interval
    disagree with each other.
    """
    n_img = len(stats["tp"])
    rng = np.random.default_rng(seed)
    point = summary_metrics(ap50_per_class(stats, range(n_img)), names, observed, excluded)
    keys = list(point)
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n_img, n_img)      # with replacement, by image
        m = summary_metrics(ap50_per_class(stats, idx), names, observed, excluded)
        for k in acc:
            if m[k] is not None:
                acc[k].append(m[k])
    out = {}
    for k, v in acc.items():
        if not v:
            out[k] = (point[k], None, None)      # explicitly empty, never fabricated
            continue
        a = np.asarray(v, dtype=float)
        out[k] = (point[k], float(np.percentile(a, 100 * alpha / 2)),
                  float(np.percentile(a, 100 * (1 - alpha / 2))))
    return out


def paired_bootstrap(stats_a, stats_b, names, observed, excluded,
                     n_boot: int, seed: int = 42, alpha: float = 0.05):
    """Paired bootstrap: the same image sample is applied to both arms.

    Returns {metric: (point_delta, lo, hi)} for B - A. The point estimate is
    the full-sample difference of the two arms - NOT the mean of the resampled
    deltas (for AP these differ; using the mean would put the table's delta
    and the interval's centre at different numbers).

    Prerequisite: the two arms must be image-order compatible (asserted by the
    caller via `order_compatible`).
    """
    n_img = len(stats_a["tp"])
    rng = np.random.default_rng(seed)
    keys = ["mAP50_clean", "mAP50"] + list(observed)
    ma0 = summary_metrics(ap50_per_class(stats_a, range(n_img)), names, observed, excluded)
    mb0 = summary_metrics(ap50_per_class(stats_b, range(n_img)), names, observed, excluded)
    point = {k: (None if (ma0[k] is None or mb0[k] is None) else mb0[k] - ma0[k]) for k in keys}
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n_img, n_img)
        ma = summary_metrics(ap50_per_class(stats_a, idx), names, observed, excluded)
        mb = summary_metrics(ap50_per_class(stats_b, idx), names, observed, excluded)
        for k in acc:
            if ma[k] is not None and mb[k] is not None:
                acc[k].append(mb[k] - ma[k])
    out = {}
    for k, v in acc.items():
        if not v or point[k] is None:
            out[k] = (point[k], None, None)
            continue
        a = np.asarray(v, dtype=float)
        out[k] = (point[k], float(np.percentile(a, 100 * alpha / 2)),
                  float(np.percentile(a, 100 * (1 - alpha / 2))))
    return out


def order_compatible(sa, sb):
    """Per-image class-set equality between two arms (the prerequisite for pairing).

    Equal image count alone proves nothing: a silently permuted order makes
    every pair a mismatch while the output still looks plausible.
    """
    if len(sa["tp"]) != len(sb["tp"]):
        return False, f"image count differs: {len(sa['tp'])} vs {len(sb['tp'])}"
    for i, (a, b) in enumerate(zip(sa["target_img"], sb["target_img"])):
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            return False, f"image {i}: class set differs ({a} vs {b})"
    return True, "per-image class sets identical"


# ---------------------------------------------------------------------------
# Self-test: synthetic data, no GPU, no dataset needed
# ---------------------------------------------------------------------------
def _mk_img(n_pred, n_gt, n_tp, cls=1, conf=0.9):
    """Synthetic per-image stats: n_pred predictions, n_gt targets, n_tp true positives."""
    tp = np.zeros((n_pred, N_IOU), dtype=np.float32)
    if n_tp:
        tp[:n_tp, 0] = 1.0
    confs = np.linspace(conf, max(conf - 0.05, 0.05), n_pred) if n_pred else np.zeros(0, dtype=np.float32)
    return {
        "tp": tp,
        "conf": confs.astype(np.float32),
        "pred_cls": np.full(n_pred, cls, dtype=np.float32),
        "target_cls": np.full(n_gt, cls, dtype=np.float32),
        "target_img": np.array([cls], dtype=np.float32),
    }


def _fake_stats(n_img=12, cls=1):
    """Reproducible synthetic per-image stats for the self-test.

    Images must NOT all be identical (TP counts alternate 1/2 here): identical
    images make every resample equal the full sample, giving zero-width
    intervals - a fixture that 'runs' but tests nothing.
    """
    imgs = [_mk_img(2, 2, 1 + (i % 2), cls=cls) for i in range(n_img)]
    return {k: [im[k] for im in imgs] for k in ("tp", "conf", "pred_cls", "target_cls", "target_img")}


def selftest() -> int:
    """CPU-only checks of the statistical core (ap_per_class is pure numpy)."""
    ok = True
    observed = ("c0", "c1", "c2")
    excluded = ("c2",)
    names = ["c0", "c1", "c2"]

    # T1: summary metrics fold correctly; a missing class must stay None (never 0).
    stats = _fake_stats(12, cls=1)
    pc = ap50_per_class(stats, range(12))
    m = summary_metrics(pc, names, observed, excluded)
    exp = pc.get(1)
    if m["c1"] != exp or m["mAP50_clean"] != exp or m["c2"] is not None:
        print(f"[FAIL] T1 summary fold: {m}")
        ok = False
    else:
        print("[PASS] T1 summary fold (absent class stays None; no fabricated zeros)")

    # T2: an arm with strictly more TPs must show a positive paired delta and
    #     the point estimate must equal the full-sample difference.
    stats_b = _fake_stats(12, cls=1)
    for i in range(12):
        stats_b["tp"][i] = stats_b["tp"][i].copy()
        stats_b["tp"][i][:, 0] = 1.0                     # every prediction correct in arm B
    res = paired_bootstrap(stats, stats_b, names, observed, excluded, n_boot=50, seed=0)
    d = res["mAP50_clean"][0]
    if not (d is not None and d > 0):
        print(f"[FAIL] T2 positive delta expected, got {d}")
        ok = False
    else:
        print(f"[PASS] T2 paired delta = {d:+.4f} (point == full-sample difference)")

    # T3: zero-width intervals must not silently appear for identical arms...
    res_same = paired_bootstrap(stats, stats, names, observed, excluded, n_boot=50, seed=0)
    lo, hi = res_same["mAP50_clean"][1], res_same["mAP50_clean"][2]
    if not (lo is None or (lo <= res_same["mAP50_clean"][0] <= hi)):
        print(f"[FAIL] T3 interval sanity: {res_same['mAP50_clean']}")
        ok = False
    else:
        print("[PASS] T3 interval brackets the point estimate")

    # T4: order compatibility must accept aligned arms and reject any arm whose
    #     per-image class sets differ. The negative fixture must actually differ:
    #     with all-equal class sets a reversal is a no-op and the check would be
    #     vacuous (a self-test that cannot fail proves nothing).
    stats_perm = _fake_stats(12, cls=1)
    stats_perm["target_img"][0] = np.array([0], dtype=np.float32)
    good, msg = order_compatible(stats, stats_b)
    bad, msg_bad = order_compatible(stats, stats_perm)
    if not good or bad:
        print(f"[FAIL] T4 order check: good={good} ({msg}) bad={bad} ({msg_bad})")
        ok = False
    else:
        print("[PASS] T4 order compatibility accepts aligned / rejects differing arms")

    # T5: metric_labels must hard-fail when a support count is missing.
    try:
        metric_labels({"c0": 3}, observed, excluded)
        print("[FAIL] T5 missing support did not raise")
        ok = False
    except SystemExit:
        print("[PASS] T5 missing support count hard-fails (no silent n=0)")

    print("selftest:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Paired per-image bootstrap CIs with a reconciliation gate")
    ap.add_argument("--data", type=str, help="dataset yaml for validation")
    ap.add_argument("--ckpt-a", type=Path, help="arm A checkpoint (e.g. supervised best.pt)")
    ap.add_argument("--ckpt-b", type=Path, help="arm B checkpoint (e.g. semi-supervised best.pt)")
    ap.add_argument("--label-a", type=str, default="arm_a")
    ap.add_argument("--label-b", type=str, default="arm_b")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=None,
                    help="dataloader workers for validation; omit to keep the checkpoint's value")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--classes", type=str, default=",".join(DEFAULT_CLASSES),
                    help="comma-separated observed class names")
    ap.add_argument("--drop-class", type=str, default=",".join(DEFAULT_EXCLUDED),
                    help="comma-separated classes excluded from mAP50_clean (leakage audit outcome)")
    ap.add_argument("--out", type=Path, default=None, help="markdown output path")
    ap.add_argument("--json", type=Path, default=None, help="json output path")
    ap.add_argument("--selftest", action="store_true", help="run the CPU-only synthetic self-test")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if not (args.data and args.ckpt_a and args.ckpt_b):
        ap.error("--data, --ckpt-a and --ckpt-b are required (or use --selftest)")
    observed = tuple(s.strip() for s in args.classes.split(",") if s.strip())
    excluded = tuple(s.strip() for s in args.drop_class.split(",") if s.strip())

    print(f"[collect] arm A = {args.label_a} ({args.ckpt_a})")
    stats_a, box_a, names, n_a = collect(args.ckpt_a, args.data, args.device, args.imgsz, args.workers)
    print(f"[collect] arm B = {args.label_b} ({args.ckpt_b})")
    stats_b, box_b, names_b, n_b = collect(args.ckpt_b, args.data, args.device, args.imgsz, args.workers)
    if names != names_b:
        raise SystemExit(f"[err] class-name order differs between arms: {names} vs {names_b}")

    ok_a, detail_a = consistency_check(stats_a, box_a, names)
    ok_b, detail_b = consistency_check(stats_b, box_b, names)
    print(f"[gate] reconciliation A: {'PASS' if ok_a else 'FAIL'} ({detail_a})")
    print(f"[gate] reconciliation B: {'PASS' if ok_b else 'FAIL'} ({detail_b})")
    if not (ok_a and ok_b):
        raise SystemExit("[err] reconciliation gate failed; refusing to emit intervals")

    compatible, msg = order_compatible(stats_a, stats_b)
    print(f"[gate] order compatibility: {'PASS' if compatible else 'FAIL'} ({msg})")
    if not compatible:
        raise SystemExit("[err] arms are not image-order compatible; pairing would be invalid")

    support = class_support(stats_a, names)
    labels = metric_labels(support, observed, excluded)   # hard-fails on missing support
    res = paired_bootstrap(stats_a, stats_b, names, observed, excluded,
                           n_boot=args.n_boot, seed=args.seed, alpha=args.alpha)

    # Per-arm full-sample point estimates (the same values each delta is built
    # from) so the JSON documents both levels, not only the difference.
    point_a = summary_metrics(ap50_per_class(stats_a, range(n_a)), names, observed, excluded)
    point_b = summary_metrics(ap50_per_class(stats_b, range(n_a)), names, observed, excluded)

    lines = [
        f"# Paired bootstrap (per-image, {args.n_boot} resamples, seed {args.seed}, "
        f"{int((1 - args.alpha) * 100)}% CI)",
        "",
        f"- arm A: {args.label_a} ({args.ckpt_a})",
        f"- arm B: {args.label_b} ({args.ckpt_b})",
        f"- images: {n_a} (order-compatible with arm B)",
        f"- reconciliation: A {'PASS' if ok_a else 'FAIL'}, B {'PASS' if ok_b else 'FAIL'} (tol {AP_TOL:g})",
        "",
        "| metric | A | B | delta (B-A) | CI low | CI high |",
        "|---|---|---|---|---|---|",
    ]
    json_out = {"arm_a": args.label_a, "arm_b": args.label_b, "n_images": n_a,
                "n_boot": args.n_boot, "seed": args.seed, "alpha": args.alpha,
                "classes": list(observed), "excluded": list(excluded), "results": {}}
    fmt = lambda v: "-" if v is None else f"{v:+.4f}"
    for k, (d, lo, hi) in res.items():
        lines.append(f"| {labels[k]} | {fmt(point_a.get(k))} | {fmt(point_b.get(k))} "
                     f"| {fmt(d)} | {fmt(lo)} | {fmt(hi)} |")
        json_out["results"][k] = {"delta": d, "ci_low": lo, "ci_high": hi, "support": support.get(k)}
    json_out["arms"] = {"A": point_a, "B": point_b}
    md = "\\n".join(lines) + "\\n"

    print()
    print(md)
    out_md = args.out or Path(f"paired_ci_{args.label_b}_vs_{args.label_a}.md")
    out_md.write_text(md, encoding="utf-8")
    out_js = args.json or out_md.with_suffix(".json")
    out_js.write_text(json.dumps(json_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out_md}\n-> {out_js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
