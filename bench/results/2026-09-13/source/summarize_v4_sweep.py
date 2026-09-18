#!/usr/bin/env python3
"""Aggregate bench_v4_sweep output.

Win/tie/loss uses the repo's ~5% reality threshold: anything inside +/-5% is a
tie, not a win. Ratios are aggregated with a geometric mean because they are
ratios; an arithmetic mean of speedups would overweight the large ones.
Rows where v4 refused the shape carry no v4 columns and are excluded from every
v4 statistic, and listed separately.
"""
import csv, sys, math

TH = 0.05

def regime(M, K, N):
    if N <= 256:                           return "narrow N (<=256)"
    if M <= 128:                           return "narrow M (<=128)"
    if M >= 1024 and K >= 1024 and N >= 1024: return "all three large"
    return "mixed"

def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else float("nan")

def wtl(rs, key):
    w = sum(1 for r in rs if r[key] > 1+TH)
    l = sum(1 for r in rs if r[key] < 1-TH)
    return w, len(rs)-w-l, l

def main(path):
    rows = list(csv.DictReader(open(path)))
    native, refused = [], []
    for r in rows:
        d = dict(tag=r["tag"], M=int(r["M"]), K=int(r["K"]), N=int(r["N"]),
                 path=r["v4_path"], packB=float(r["packB_MiB"]), packA=float(r["packA_MiB"]))
        if r["v4_path"] != "native" or not r["v4_gflops"]:
            refused.append(d); continue
        d.update(v3=float(r["v3_gflops"]), v4=float(r["v4_gflops"]),
                 acc=float(r["acc_gflops"]), mp=float(r["mp_gflops"]),
                 v4_v3=float(r["v4_over_v3"]), v4_mp=float(r["v4_over_mp"]),
                 v4_acc=float(r["v4_over_acc"]),
                 sp=max(float(r["v3_spread"]), float(r["v4_spread"])),
                 err=float(r["relerr_v4"]))
        d["regime"] = regime(d["M"], d["K"], d["N"])
        native.append(d)

    print(f"{len(native)} shapes ran natively on v4, {len(refused)} refused.\n")
    if refused:
        print("Refused by v4 (no v4 number; NOT replaced by a fallback):")
        for d in refused: print(f"  {d['tag']:12} {d['M']}x{d['K']}x{d['N']:<8} {d['path']}")
        print()

    print("Win / tie / loss, +/-5% threshold")
    print(f"{'':22}{'win':>5}{'tie':>5}{'loss':>6}   geomean")
    for key, name in (("v4_v3","v4 vs v3"), ("v4_mp","v4 vs MpGEMM"), ("v4_acc","v4 vs Accelerate")):
        w,t,l = wtl(native, key)
        print(f"  {name:20}{w:>5}{t:>5}{l:>6}   {geomean([r[key] for r in native]):.3f}x")
    print()

    print("By regime  (geomean of the ratio; win/tie/loss vs v3)")
    print(f"{'regime':22}{'n':>3}  {'v4/v3':>8}{'v4/MpGEMM':>11}{'v4/Accel':>10}   w/t/l vs v3")
    for g in ("narrow N (<=256)", "narrow M (<=128)", "all three large", "mixed"):
        rs = [r for r in native if r["regime"] == g]
        if not rs: continue
        w,t,l = wtl(rs, "v4_v3")
        print(f"{g:22}{len(rs):>3}  {geomean([r['v4_v3'] for r in rs]):>7.3f}x"
              f"{geomean([r['v4_mp'] for r in rs]):>10.3f}x{geomean([r['v4_acc'] for r in rs]):>9.3f}x"
              f"   {w}/{t}/{l}")
    print()

    s = sorted(native, key=lambda r: r["v4_v3"])
    print("Most regressed vs v3")
    for r in s[:5]:
        print(f"  {r['tag']:12} {r['M']}x{r['K']}x{r['N']:<8} v4/v3 {r['v4_v3']:.3f}x   "
              f"v3 {r['v3']:.0f} -> v4 {r['v4']:.0f} GFLOP/s   packB {r['packB']:.0f} MiB")
    print("Most improved vs v3")
    for r in reversed(s[-5:]):
        print(f"  {r['tag']:12} {r['M']}x{r['K']}x{r['N']:<8} v4/v3 {r['v4_v3']:.3f}x   "
              f"v3 {r['v3']:.0f} -> v4 {r['v4']:.0f} GFLOP/s   packB {r['packB']:.0f} MiB")
    print()

    worst_err = max(native, key=lambda r: r["err"])
    worst_sp  = max(native, key=lambda r: r["sp"])
    print(f"Largest v4 relative error vs v3 : {worst_err['err']:.3e}  ({worst_err['tag']})")
    print(f"Largest run-to-run spread        : {worst_sp['sp']:.2f}%  ({worst_sp['tag']})")
    print(f"Largest packed B                 : {max(r['packB'] for r in native):.0f} MiB "
          f"({max(native, key=lambda r: r['packB'])['tag']})")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench/results/2026-09-13/v4_sweep.csv")
