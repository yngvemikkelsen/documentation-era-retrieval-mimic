#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
Did every cell actually run, with the settings it was asked for?

    uv run paper21c_verify_v1.py \
        --check "path=~/projects/paper21c_v8,model=sentence-transformers/all-mpnet-base-v2,min_year=2013,scrub=0,pool=1000,cells=contrastive:mimic3;contrastive:mimic4"

Seconds. Reads cell files only. Exits non-zero if anything is wrong.

WHY THIS EXISTS
---------------
paper21c_crossover_v5.py returns 0 even when cells fail. Its main() collects
failures into a list, prints "CELLS THAT FAILED", and then returns 0 anyway. So
a run script that tests the exit code learns nothing: a queue can lose three of
nine cells and report success.

It also decides to skip a cell with set(want).issubset(set(done)) -- checkpoint
steps only. It never compares --model, --mimic4-min-year, --train-pool or
--pool. A PubMedBERT run into a workdir holding mpnet cells prints
"SKIP: already complete" and hands back the mpnet numbers under the PubMedBERT
label. A clean workdir prevents it; this script detects it if it happened
anyway, by reading what each cell says it was.

WHAT IS CHECKED PER CELL
------------------------
  exists              the file is there at all
  partial == False    the final save ran; a partial cell is a crashed cell
  collapsed_at_step   null, or the cell is a representation collapse
  model               matches, which is what catches a silent skip
  mimic4_min_year     matches the restriction that was requested
  scrub_deid          matches
  pool and seed       match
  n per era           equals pool, so no era quietly evaluated fewer queries
  steps_done          contains every requested checkpoint

WHAT CANNOT BE CHECKED HERE
---------------------------
--train-pool is NOT recorded in the cell metadata. The only record of it is the
log line "building training pairs from N held-out notes". Pass --logdir and the
log is grepped for that line; without it, training volume is unverified and the
script says so rather than implying it checked.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

TAKE_RE = re.compile(r"building training pairs from ([\d,]+) held-out notes")


def parse_check(spec: str) -> dict:
    d = {}
    for part in spec.split(","):
        if "=" not in part:
            raise SystemExit(f"bad --check fragment {part!r}; expected key=value")
        k, v = part.split("=", 1)
        d[k.strip()] = v.strip()
    for k in ("path", "cells"):
        if k not in d:
            raise SystemExit(f"--check needs {k}=; got {sorted(d)}")
    return d


def main() -> int:
    p = argparse.ArgumentParser(description="Verify produced cells against request")
    p.add_argument("--check", action="append", default=[],
                   help="path=...,model=...,min_year=...,scrub=0|1,pool=N,"
                        "steps=a;b;c,cells=obj:train;obj:train")
    p.add_argument("--logdir", type=Path, default=None,
                   help="where the run logs are, so --train-pool can be verified")
    p.add_argument("--expect-train-pool", default=None,
                   help="per-log expectations, as stem=N pairs: "
                        "'v8_main=18000,v8_within=7000'. One global number will "
                        "not do: the within-MIMIC-IV control is compared against "
                        "itself and legitimately uses a smaller pool, so a single "
                        "value would flag it every time. A log with no entry here "
                        "is reported as unchecked, never as passing.")
    p.add_argument("--allow-missing", action="store_true",
                   help="resume mode: a cell not yet on disk is reported PENDING "
                        "instead of failing, but every cell that IS on disk must "
                        "still match the requested configuration. This is what "
                        "makes restarting safe: the crossover script skips a cell "
                        "on checkpoint steps alone, so resuming into a workdir is "
                        "only safe if the cells already there were run with the "
                        "same model, restriction, pool and seed.")
    p.add_argument("--skip-logs", action="store_true",
                   help="omit the training-pool check, for a resume check run "
                        "before any log exists")
    a = p.parse_args()

    if not a.check:
        raise SystemExit("nothing to verify; pass at least one --check")

    bad: list[str] = []
    rows: list[tuple] = []

    for spec in a.check:
        c = parse_check(spec)
        wd = Path(c["path"]).expanduser()
        want_steps = [int(x) for x in c.get("steps", "250;500;1000;2500").split(";")
                      if x.strip()]
        want_pool = int(c["pool"]) if "pool" in c else None
        want_scrub = c.get("scrub")
        tag = "_scrubbed" if want_scrub == "1" else ""
        for cell in c["cells"].split(";"):
            if not cell.strip():
                continue
            obj, train_on = cell.split(":")
            f = wd / f"cell_{obj}_{train_on}{tag}.npz"
            if not f.exists():
                if a.allow_missing:
                    rows.append((f.name, wd.name, "PENDING", "", "", ""))
                else:
                    bad.append(f"MISSING  {f}")
                    rows.append((f.name, wd.name, "MISSING", "", "", ""))
                continue
            try:
                z = np.load(f, allow_pickle=True)
                m = json.loads(str(z["meta"].item()))
            except Exception as e:
                bad.append(f"UNREADABLE  {f}  ({type(e).__name__})")
                rows.append((f.name, wd.name, "UNREADABLE", "", "", ""))
                continue

            probs = []
            if m.get("partial", False):
                if a.allow_missing:
                    rows.append((f.name, wd.name, "PARTIAL", "", "", ""))
                    continue
                probs.append("partial=True (cell crashed before the final save)")
            if m.get("collapsed_at_step"):
                probs.append(f"COLLAPSED at step {m['collapsed_at_step']:,}")
            for key, want in (("model", c.get("model")),
                              ("mimic4_min_year", c.get("min_year")),
                              ("pool", c.get("pool")),
                              ("seed", c.get("seed"))):
                if want is None:
                    continue
                got = m.get(key)
                wantv = None if want in ("", "None", "none") else want
                if str(got) != str(wantv):
                    probs.append(f"{key}={got!r} but {wantv!r} was requested")
            if want_scrub is not None:
                if bool(m.get("scrub_deid")) != (want_scrub == "1"):
                    probs.append(f"scrub_deid={m.get('scrub_deid')} but "
                                 f"{want_scrub} was requested")
            missing = sorted(set(want_steps) - set(m.get("steps_done", [])))
            if missing:
                probs.append(f"missing checkpoints {missing}")

            ns = {}
            for k in z.files:
                if k.startswith("rr__"):
                    _, s, e = k.split("__")
                    ns.setdefault(e, set()).add(len(z[k]))
            for e, sizes in sorted(ns.items()):
                if len(sizes) != 1:
                    probs.append(f"{e} has inconsistent n across steps {sorted(sizes)}")
                elif want_pool is not None and next(iter(sizes)) != want_pool:
                    probs.append(f"{e} n={next(iter(sizes))} but pool={want_pool}")

            rows.append((f.name, wd.name, "OK" if not probs else "BAD",
                         str(m.get("model", ""))[-22:],
                         str(m.get("mimic4_min_year")),
                         ",".join(str(x) for x in sorted(ns))))
            for pr in probs:
                bad.append(f"{f.name}: {pr}")

    print("=" * 104)
    print("CELL VERIFICATION")
    print("=" * 104)
    print(f"  {'cell':<40}{'workdir':<22}{'state':<11}{'model':<24}{'min_yr':>7}")
    for name, wdn, st, mdl, my, _eras in rows:
        print(f"  {name:<40}{wdn:<22}{st:<11}{mdl:<24}{my:>7}")

    print("\n" + "=" * 104)
    print("TRAINING-POOL VERIFICATION")
    print("=" * 104)
    if a.skip_logs:
        print("  SKIPPED — resume check, before any log exists")
    elif a.logdir is None:
        print("  NOT CHECKED — no --logdir given. --train-pool is not recorded in")
        print("  the cell metadata, so training volume is unverified.")
    else:
        want: dict[str, int] = {}
        for frag in (a.expect_train_pool or "").split(","):
            if frag.strip():
                k, v = frag.split("=", 1)
                want[k.strip()] = int(v)
        found = False
        print(f"  {'log':<24}{'notes trained on':<22}{'expected':>10}{'state':>12}")
        for lg in sorted(Path(a.logdir).expanduser().glob("v8_*.log")):
            takes = sorted({int(x.replace(",", ""))
                            for x in TAKE_RE.findall(lg.read_text(errors="replace"))})
            if not takes:
                continue
            found = True
            exp = want.get(lg.stem)
            if exp is None:
                state, note = "UNCHECKED", f"{lg.name}: no expected train-pool given"
            elif takes == [exp]:
                state, note = "OK", None
            else:
                state = "MISMATCH"
                note = (f"{lg.name}: trained on {takes} notes, expected {exp:,} "
                        "for every arm in this configuration")
            print(f"  {lg.name:<24}{str(takes):<22}"
                  f"{('-' if exp is None else format(exp, ',')):>10}{state:>12}")
            if note:
                bad.append(note)
        if not found:
            print("  no 'building training pairs from N held-out notes' line in any")
            print("  v8_*.log; training volume is unverified.")
            bad.append("no training-pool line found in the logs")

    print("\n" + "=" * 104)
    if bad:
        print(f"VERIFICATION FAILED — {len(bad)} problem(s)")
        print("=" * 104)
        for b in bad:
            print(f"  {b}")
        print("\n  Do not report any number from these cells until every line above")
        print("  is resolved. A model or min_year mismatch means a silent SKIP:")
        print("  delete that workdir and run the cell again.")
        return 2
    pend = sum(1 for r in rows if r[2] in ("PENDING", "PARTIAL"))
    if a.allow_missing:
        print(f"RESUME IS SAFE — {len(rows) - pend} cell(s) on disk match the "
              f"requested configuration, {pend} still to run")
        print("=" * 104)
        return 0
    print("VERIFICATION PASSED — every cell ran with the settings requested")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
