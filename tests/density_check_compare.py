"""Step 2 of the density check by eye: compare your answers with the program.

Usage (12 letters, in picture order; L = Low, M = Medium, H = High):
    python tests\\density_check_compare.py L M M H ...
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
file = PROJECT_ROOT / "output" / "density_check" / "program_status.json"
if not file.exists():
    sys.exit("Run tests\\density_check_frames.py first.")
data = json.loads(file.read_text())

names = {"L": "Low", "M": "Medium", "H": "High"}
order = ["Low", "Medium", "High"]
answers = [a.upper() for a in sys.argv[1:]]
if len(answers) != len(data) or any(a not in names for a in answers):
    sys.exit(f"Give {len(data)} letters, each L, M or H, in picture order.")

matrix = {(mine, prog): 0 for mine in order for prog in order}
exact = within_one = 0
print(f"{'#':>2} {'time':>6}  {'you':<7} {'program':<7} {'occupancy':>9}  result")
for row, letter in zip(data, answers):
    mine, prog = names[letter], row["status"]
    diff = abs(order.index(mine) - order.index(prog))
    exact += diff == 0
    within_one += diff <= 1
    matrix[(mine, prog)] += 1
    verdict = "match" if diff == 0 else ("off by one" if diff == 1 else "OFF BY TWO")
    print(f"{row['n']:>2} {row['time']:>5.1f}s  {mine:<7} {prog:<7} {row['occupancy'] * 100:>8.1f}%  {verdict}")

total = len(data)
print(f"\nExact match: {exact} of {total} ({exact / total:.0%})")
print(f"Within one level: {within_one} of {total} ({within_one / total:.0%})")
print("\nWhat you said -> what the program said")
for mine in order:
    print(f"  {mine:<7}: " + ", ".join(f"{prog} {matrix[(mine, prog)]}" for prog in order))