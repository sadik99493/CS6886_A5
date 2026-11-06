#!/usr/bin/env python3
"""
run_all.py -- time every variant on every input in ONE session and write results.csv.

Steps:
  1. make clean, then make naive loops tiled fused best omp (omp uses all 8 threads)
  2. make check
  3. omp thread sweep (1, 2, 4, 6, 8 threads) on every input; each sweep output is
     checked against naive with the same rule as `make check` (rel. error <= 1e-4)
  4. write results.csv (time, GFLOP/s, modelled arithmetic intensity, check status)
     and save the raw output to report/final_run.log

Usage (from the repo root):  python3 scripts/run_all.py
"""
import glob
import os
import re
import struct
import subprocess
from array import array

VARIANTS = ["naive", "loops", "tiled", "fused", "best"]
THREADS = [1, 2, 4, 6, 8]
LINE = re.compile(r"^(\w+)\s+N=(\d+)\s+d=(\d+)\s+threads=(\d+)\s+time_ms=\s*([\d.]+)\s+GFLOP/s=\s*([\d.]+)")
log = []


def run(cmd, env=None):
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                         env={**os.environ, **(env or {})})
    text = out.stdout + out.stderr
    log.append("$ " + cmd + "\n" + text)
    print(text, end="")
    return text


def parse(text):
    """-> {(variant, N, d, threads): (time_ms, gflops)}"""
    res = {}
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if m:
            v, n, d, t, ms, g = m.groups()
            res[(v, int(n), int(d), int(t))] = (float(ms), float(g))
    return res


def load(path):
    with open(path, "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        a = array("f")
        a.fromfile(f, n * d)
    return a


def rel_error(path, ref_path):
    out, ref = load(path), load(ref_path)
    scale = max(abs(x) for x in ref)
    return max(abs(a - b) for a, b in zip(out, ref)) / scale


def intensity(variant, n, d):
    """Modelled FLOPs / DRAM bytes (see report): S traffic only if S is stored."""
    flops = 4.0 * n * n * d
    compulsory = 16.0 * n * d  # read Q, K, V + write O, 4 B each
    s_traffic = 16.0 * n * n if variant in ("naive", "loops", "tiled") else 0.0
    return flops / (compulsory + s_traffic)


def main():
    inputs = sorted(glob.glob("input/*.bin"))
    run("make clean")
    results = parse(run("make naive loops tiled fused best omp"))
    check_text = run("make check")
    status = {}
    for line in check_text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] in ("PASS", "FAIL"):
            status[(parts[1], parts[0])] = parts[2]  # (variant, input file)

    # omp thread sweep, each output verified against naive
    sweep_status = {}
    for f in inputs:
        base = os.path.basename(f)
        for t in THREADS:
            out_path = "output/omp_sweep_t%d_%s" % (t, base)
            results.update(parse(run("./build/omp %s %s" % (f, out_path),
                                     env={"OMP_NUM_THREADS": str(t)})))
            err = rel_error(out_path, "output/naive_" + base)
            sweep_status[(base, t)] = "PASS" if err <= 1e-4 else "FAIL"
            os.remove(out_path)

    rows = ["variant,N,d,threads,time_ms,gflops,arithmetic_intensity_flop_per_byte,make_check"]
    shapes = []
    for f in inputs:
        with open(f, "rb") as fh:
            shapes.append((os.path.basename(f),) + struct.unpack("<ii", fh.read(8)))
    order = {(1024, 64): 0, (2048, 64): 1, (4096, 64): 2, (4096, 128): 3}
    shapes.sort(key=lambda s: order.get((s[1], s[2]), 99))
    for v in VARIANTS + ["omp"]:
        for base, n, d in shapes:
            for t in (THREADS if v == "omp" else [1]):
                ms, g = results.get((v, n, d, t), (None, None))
                if v == "naive":
                    chk = "reference"
                elif v == "omp":
                    chk = sweep_status.get((base, t), "")
                else:
                    chk = status.get((v, base), "")
                rows.append("%s,%d,%d,%d,%s,%s,%.2f,%s" % (
                    v, n, d, t, "%.3f" % ms if ms else "", "%.2f" % g if g else "",
                    intensity(v, n, d), chk))
    with open("results.csv", "w", newline="") as fh:
        fh.write("\n".join(rows) + "\n")
    with open("report/final_run.log", "w") as fh:
        fh.write("\n".join(log))
    print("\nwrote results.csv (%d rows) and report/final_run.log" % (len(rows) - 1))


if __name__ == "__main__":
    main()
