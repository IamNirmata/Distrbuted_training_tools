#!/usr/bin/env python3
"""
All-pairs round scheduler for parallel pairwise tests (e.g., AllReduce).

Properties:
- Covers every unordered pair exactly once (complete graph K_n 1-factorization).
- No node appears more than once per round (pairs in a round can run in parallel).
- Even n  -> n-1 rounds, each with n/2 pairs.
- Odd  n  -> n rounds (one BYE each round; BYE omitted from output).

Usage examples:
  # 80 nodes labeled 0..79
  python3 generate_permutations.py --nitems 80 --verify --format text | head

  # With node list file (one host/IP per line) and CSV output
  python3 generate_permutations.py --nodes-file nodes.txt --verify --format csv > rounds.csv
"""

from __future__ import annotations
import argparse
import json
import signal
import sys
from itertools import combinations
from typing import List, Tuple, Optional

Pair = Tuple[str, str]

# --------------------------- core scheduler ---------------------------

def schedule_round_robin(items: List[str]) -> List[List[Pair]]:
    """
    Returns a list of rounds; each round is a list of disjoint (a,b) pairs.
    Implements the classic 'circle method' rotation.
    """
    odd = (len(items) % 2 == 1)
    BYE: Optional[str] = None
    arr = items[:] + ([BYE] if odd else [])
    n = len(arr)
    if n < 2:
        return []

    total_rounds = n - 1  # works for both even/odd (odd has a BYE appended)
    half = n // 2

    rounds: List[List[Pair]] = []
    for _ in range(total_rounds):
        left = arr[:half]
        right = arr[half:][::-1]

        rnd: List[Pair] = []
        for a, b in zip(left, right):
            if a is not None and b is not None:  # drop BYE pairs
                rnd.append((str(a), str(b)))
        rounds.append(rnd)

        # rotate all but the first element:
        # [a0, a1, a2, ..., a_{n-2}, a_{n-1}] -> [a0, a_{n-1}, a1, a2, ..., a_{n-2}]
        if n > 2:
            arr = [arr[0], arr[-1], *arr[1:-1]]

    return rounds

# ----------------------------- verify --------------------------------

def verify_rounds(rounds: List[List[Pair]], items: List[str]) -> bool:
    """
    Verifies:
      1) Per-round uniqueness: no node repeats within a round.
      2) Global coverage: every unordered pair appears exactly once.
    Prints results and returns True on success, False otherwise.
    """
    ok = True

    # 1) per-round uniqueness
    for i, rnd in enumerate(rounds):
        used = set()
        for a, b in rnd:
            if a in used or b in used:
                print(f"[VERIFY] Round {i} repeats node in pair {(a, b)}")
                ok = False
            used.add(a)
            used.add(b)

    # 2) coverage: unordered pairs
    want = set(combinations(items, 2))  # unordered target
    seen = set()
    for rnd in rounds:
        for a, b in rnd:
            seen.add(tuple(sorted((a, b))))  # canonicalize to unordered

    if seen != want:
        missing = want - seen
        extra = seen - want
        print(f"[VERIFY] Coverage mismatch: expected {len(want)} pairs, got {len(seen)}")
        if missing:
            mlist = list(missing)[:10]
            print(f"[VERIFY] Missing {len(missing)} pairs (first 10): {mlist}")
        if extra:
            elist = list(extra)[:10]
            print(f"[VERIFY] Extra {len(extra)} pairs (first 10): {elist}")
        ok = False
    else:
        print(f"Coverage: {len(seen)}/{len(want)} unordered pairs -> OK")

    return ok

# ---------------------------- I/O utils -------------------------------

def load_items(n: Optional[int], nodes_file: Optional[str]) -> List[str]:
    if nodes_file:
        with open(nodes_file, "r", encoding="utf-8") as f:
            items = [ln.strip() for ln in f if ln.strip()]
        return items
    if n is None or n < 2:
        sys.exit("Provide --nodes-file or --nitems >= 2")
    return [str(i) for i in range(n)]

def emit(rounds: List[List[Pair]], fmt: str) -> None:
    if fmt == "text":
        for rnd in rounds:
            print(f'        "' + " | ".join(f"{a} {b}" for a, b in rnd) + '"')
    elif fmt == "csv":
        print("round,a,b")
        for i, rnd in enumerate(rounds):
            for a, b in rnd:
                print(f"{i},{a},{b}")
    elif fmt == "jsonl":
        for i, rnd in enumerate(rounds):
            print(json.dumps({"round": i, "pairs": rnd}))
    else:
        sys.exit(f"Unknown --format '{fmt}'")


# ------------------------------ main ---------------------------------

def main():
    # Make piping to `head`/`sed` etc. not crash the script with BrokenPipeError.
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    ap = argparse.ArgumentParser(
        description="Generate parallel-safe all-pairs rounds (no node repeats per round)."
    )
    ap.add_argument("--nodes-file", type=str, default=None, help="Path with one node per line.")
    ap.add_argument("--nitems", type=int, default=None, help="Use nodes 0..n-1 (ignored if --nodes-file).")
    ap.add_argument("--format", choices=["text", "csv", "jsonl"], default="text", help="Output format.")
    ap.add_argument("--verify", action="store_true", help="Verify per-round uniqueness and global coverage.")
    args = ap.parse_args()

    items = load_items(args.nitems, args.nodes_file)
    rounds = schedule_round_robin(items)

    if args.verify:
        verify_rounds(rounds, items)

    try:
        emit(rounds, args.format)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            pass

if __name__ == "__main__":
    main()
