"""Run all automatic tests and write a report.

Usage:
    python tests\\run_all_tests.py

These tests need no video and no model (they check the logic of each module).
The report is also saved in output\\test_report.txt, so it can be copied into
the project report.
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TESTS = [
    ("Traffic density: occupancy, status, hold time", "tests/test_traffic_density.py"),
    ("Signal controller: cycle, adaptive green, emergency priority", "tests/test_signal_controller.py"),
    ("Violation rules: wrong way, stopped, red light", "tests/test_violation_detector.py"),
    ("Database: store, summary, CSV export", "tests/test_database.py"),
    ("Vehicle counter: crossing, ID bridging, duplicate filter", "tests/test_counter.py"),
    ("Demo e-challan: database, evidence picture, PDF", "tests/test_challan_demo.py"),
]

lines = []


def out(text=""):
    print(text, flush=True)
    lines.append(text)


out(f"Automatic tests, {time.strftime('%Y-%m-%d %H:%M')}")
out("-" * 88)
out(f"{'Test':<62} {'Asserts':>7} {'Time':>7}  Result")
out("-" * 88)

failed = []
total_asserts = 0
for name, script in TESTS:
    path = PROJECT_ROOT / script
    asserts = sum(1 for line in path.read_text(encoding="utf-8").splitlines()
                                    if line.strip().startswith(("assert ", "check(")))
    total_asserts += asserts
    start = time.time()
    run = subprocess.run([sys.executable, str(path)], cwd=PROJECT_ROOT, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    seconds = time.time() - start
    ok = run.returncode == 0
    out(f"{name:<62} {asserts:>7} {seconds:>6.1f}s  {'PASS' if ok else 'FAIL'}")
    if not ok:
        failed.append((name, (run.stdout + run.stderr).strip().splitlines()[-12:]))

out("-" * 88)
out(f"{len(TESTS) - len(failed)} of {len(TESTS)} test files passed ({total_asserts} assert statements).")
for name, tail in failed:
    out(f"\nFAILED: {name}")
    for line in tail:
        out("   " + line)

report = PROJECT_ROOT / "output" / "test_report.txt"
report.parent.mkdir(exist_ok=True)
report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nReport saved: {report}")
sys.exit(1 if failed else 0)