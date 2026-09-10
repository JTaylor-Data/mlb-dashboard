"""
Deception+ pipeline, BATCH DRIVER - clean rewrite (v4).
Usage (venv python): python d4_run.py <videos_dir> <out_dir> [--keep-videos]
"""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d4_extract import load_models
from d4_pipeline import process_one

TEST5 = ["skenes", "wheeler", "rogers", "yamamoto", "skubal"]


def print_timing_summary(results):
    totals, counts = {}, {}
    for r in results:
        for stage, secs in r.get("timings", {}).items():
            totals[stage] = totals.get(stage, 0.0) + secs
            counts[stage] = counts.get(stage, 0) + 1
    if not totals:
        return
    grand = sum(totals.values())
    print(f"\n{'STAGE':32s} {'TOTAL s':>9s} {'AVG s':>8s} {'% OF TOTAL':>11s} {'N':>4s}")
    for stage, total in sorted(totals.items(), key=lambda kv: -kv[1]):
        n = counts[stage]
        print(f"{stage:32s} {total:9.2f} {total/n:8.2f} {100*total/grand:10.1f}% {n:4d}")
    print(f"{'TOTAL':32s} {grand:9.2f}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    keep = "--keep-videos" in sys.argv
    videos_dir, out_dir = args[0], args[1]
    os.makedirs(out_dir, exist_ok=True)

    print("loading models...", flush=True)
    t_load = time.time()
    models = load_models()
    print(f"models loaded ({time.time()-t_load:.1f}s)", flush=True)

    results = []
    t0 = time.time()
    for tag in TEST5:
        vp = os.path.join(videos_dir, f"{tag}_0.mp4")
        if not os.path.exists(vp):
            print(f"{tag}: MISSING video ({vp})", flush=True)
            continue
        t1 = time.time()
        try:
            r = process_one(vp, models, tag, keep_video=keep,
                            landmarks_dir=os.path.join(out_dir, "landmarks"))
        except Exception:
            r = {"tag": tag, "ok": False, "stage": "exception", "error": traceback.format_exc(limit=5),
                 "timings": {}}
        elapsed = time.time() - t1
        results.append(r)
        if r["ok"]:
            pt = r["phase_times"]
            print(f"{tag}: OK [{r['method']}]  set={pt['coming_set']:.2f} start={pt['leg_lift_start']:.2f} "
                  f"peak={pt['leg_lift_peak']:.2f} plant={pt['foot_plant']:.2f} rel={pt['release']:.2f}  "
                  f"({elapsed:.1f}s)", flush=True)
        else:
            print(f"{tag}: FAILED [{r.get('method', r.get('stage'))}] - {r.get('error')}  ({elapsed:.1f}s)",
                  flush=True)

    print_timing_summary(results)
    out = {"results": results, "n_ok": sum(1 for r in results if r["ok"]), "n_total": len(results),
           "elapsed_s": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDONE {out['n_ok']}/{out['n_total']} ok in {out['elapsed_s']}s -> {out_dir}")


if __name__ == "__main__":
    main()
