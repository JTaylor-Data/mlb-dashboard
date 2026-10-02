"""
Post-training finalization script — run after train_stuff_v4.py completes.

Steps:
1. Run model_correlation_heatmap.py
2. Generate STUFF_V4_REPORT.md and STUFF_PLUS_V4_FINAL_METHODOLOGY.html
3. Methodology archival:
   - Move Methodology/in_progress/ -> Methodology/archive/in_progress/
   - Move Methodology/Old Diagnostics/ -> Methodology/archive/old_diagnostics/
   - Move Methodology/live_models/STUFF_PLUS_V3_METHODOLOGY.html -> Methodology/archive/
4. Git commit and push stuff-v4 branch
5. Tag post-stuff-v4
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DIAG_DIR = ROOT / "diagnostics" / "stuff_v4"
METHODOLOGY_DIR = ROOT / "Methodology"
LIVE_MODELS_DIR = METHODOLOGY_DIR / "live_models"
ARCHIVE_DIR = METHODOLOGY_DIR / "archive"


def run(cmd, **kwargs):
    print(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, **kwargs)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print("STDERR:", result.stderr.strip())
    return result


def step1_heatmap():
    print("\n=== Step 1: Model Correlation Heatmap ===")
    result = run([sys.executable, "tools/model_correlation_heatmap.py",
                  "--output-dir", str(DIAG_DIR)])
    return result.returncode == 0


def step2_report():
    print("\n=== Step 2: Generate Report ===")
    result = run([sys.executable, str(DIAG_DIR / "generate_report.py")])
    return result.returncode == 0


def step3_archive():
    print("\n=== Step 3: Methodology Archival ===")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Move in_progress/
    in_progress = METHODOLOGY_DIR / "in_progress"
    if in_progress.exists():
        dest = ARCHIVE_DIR / "in_progress"
        if dest.exists():
            # Merge by moving files individually
            for f in in_progress.iterdir():
                shutil.move(str(f), str(dest / f.name))
            in_progress.rmdir()
        else:
            shutil.move(str(in_progress), str(dest))
        print(f"Moved {in_progress} -> {dest}")
    else:
        print(f"Note: {in_progress} not found, skipping")

    # Move Old Diagnostics/
    old_diag = METHODOLOGY_DIR / "Old Diagnostics"
    if old_diag.exists():
        dest = ARCHIVE_DIR / "old_diagnostics"
        if dest.exists():
            for f in old_diag.iterdir():
                shutil.move(str(f), str(dest / f.name))
            old_diag.rmdir()
        else:
            shutil.move(str(old_diag), str(dest))
        print(f"Moved {old_diag} -> {dest}")
    else:
        print(f"Note: {old_diag} not found, skipping")

    # Move STUFF_PLUS_V3_METHODOLOGY.html -> archive/
    v3_html = LIVE_MODELS_DIR / "STUFF_PLUS_V3_METHODOLOGY.html"
    if v3_html.exists():
        dest = ARCHIVE_DIR / "STUFF_PLUS_V3_METHODOLOGY.html"
        shutil.move(str(v3_html), str(dest))
        print(f"Archived {v3_html} -> {dest}")
    else:
        print(f"Note: {v3_html} not found, skipping")

    # Verify other live model files are untouched
    for fname in ["LOCATION_PLUS_V2_METHODOLOGY.html", "TUNNELING_PLUS_V2_METHODOLOGY.html",
                   "DECEPTION_PLUS_V1_METHODOLOGY.html"]:
        p = LIVE_MODELS_DIR / fname
        if p.exists():
            print(f"  OK: {fname} still in live_models/")
        else:
            print(f"  WARNING: {fname} not found in live_models/ — should be there")


def step4_git():
    print("\n=== Step 4: Git commit and push ===")
    run(["git", "add", "-A"])
    run(["git", "commit", "-m",
         "feat(stuff-v4): complete Stuff+ v4 training, validation, archival\n\n"
         "- Part 1: data audit, HAA residual leakage checks\n"
         "- Part 2: ablation on 30% pitcher subsample (5-fold CV)\n"
         "- Part 3: final model trained on 2016-2026 excl 2020\n"
         "- Part 4: validation battery\n"
         "- Part 5: model correlation heatmap tool\n"
         "- Part 6: deliverables\n"
         "- Archival: in_progress/ and Old Diagnostics/ -> archive/\n"
         "- STUFF_PLUS_V3_METHODOLOGY.html -> archive/\n"
         "- STUFF_PLUS_V4_FINAL_METHODOLOGY.html added to live_models/\n\n"
         "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"])
    run(["git", "tag", "post-stuff-v4"])
    push_result = run(["git", "push", "origin", "stuff-v4"])
    run(["git", "push", "origin", "post-stuff-v4"])
    return push_result.returncode == 0


if __name__ == "__main__":
    ok1 = step1_heatmap()
    ok2 = step2_report()
    step3_archive()
    ok4 = step4_git()
    print("\nDone.")
    print(f"Heatmap: {'OK' if ok1 else 'FAILED'}")
    print(f"Report: {'OK' if ok2 else 'FAILED'}")
    print(f"Git push: {'OK' if ok4 else 'FAILED'}")
