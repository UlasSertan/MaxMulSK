#!/usr/bin/env python3
"""Summarise bench_final5 output.

The two v4 profiles are reported SEPARATELY throughout. Taking the better of the
two per row would describe a dispatch that does not exist; no automatic selection
has been implemented.

Win/tie/loss uses the repo's +/-5% threshold. Ratios are aggregated with a
geometric mean, shapes equally weighted.
"""
import csv, sys, math, collections

TH = 0.05

def geomean(xs):
    xs=[x for x in xs if x and x>0]
    return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else float("nan")

def regime(M,K,N):
    if N<=256: return "narrow N (<=256)"
    if M<=128: return "narrow M (<=128)"
    if M>=1024 and K>=1024 and N>=1024: return "all three large"
    return "mixed"

def wtl(xs):
    w=sum(1 for x in xs if x>1+TH); l=sum(1 for x in xs if x<1-TH)
    return w, len(xs)-w-l, l

def main(path):
    rows=[]
    for r in csv.DictReader(open(path)):
        d={k:(float(v) if k not in ("tag",) else v) for k,v in r.items()}
        d["M"],d["K"],d["N"]=int(d["M"]),int(d["K"]),int(d["N"])
        d["regime"]=regime(d["M"],d["K"],d["N"])
        rows.append(d)
    print(f"{len(rows)} shapes\n")

    worst_err=max(max(r["err_v4a"],r["err_v4b"]) for r in rows)
    print(f"Correctness: largest v4 relative Frobenius error vs v3 = {worst_err:.3e}")
    print(f"             largest for MpGEMM {max(r['err_mp'] for r in rows):.3e}, "
          f"Accelerate {max(r['err_acc'] for r in rows):.3e}\n")

    print("Per shape  (GFLOP/s, spread%)")
    hdr=f"{'shape':12}{'MxKxN':>20}{'v3':>14}{'v4a 16/512/2048':>18}{'v4b 32/1024/2048':>19}{'MpGEMM':>14}{'Accel':>14}"
    print(hdr)
    for r in rows:
        print(f"{r['tag']:12}{f'{r[chr(77)]}x{r[chr(75)]}x{r[chr(78)]}':>20}"
              f"{r['v3_gf']:>8.0f} {r['v3_sp']:>4.1f}"
              f"{r['v4a_gf']:>12.0f} {r['v4a_sp']:>4.1f}"
              f"{r['v4b_gf']:>13.0f} {r['v4b_sp']:>4.1f}"
              f"{r['mp_gf']:>8.0f} {r['mp_sp']:>4.1f}"
              f"{r['acc_gf']:>8.0f} {r['acc_sp']:>4.1f}")
    print()

    print("Overall, geometric mean of the ratio, and win/tie/loss at +/-5%")
    print(f"{'':28}{'geomean':>9}   win/tie/loss")
    for key,name in (("v4a_v3","v4a vs v3"),("v4a_mp","v4a vs MpGEMM"),("v4a_acc","v4a vs Accelerate"),
                     ("v4b_v3","v4b vs v3"),("v4b_mp","v4b vs MpGEMM"),("v4b_acc","v4b vs Accelerate")):
        xs=[r[key] for r in rows]; w,t,l=wtl(xs)
        print(f"  {name:26}{geomean(xs):>8.3f}x   {w}/{t}/{l}")
    xs=[r["v4b_gf"]/r["v4a_gf"] for r in rows]; w,t,l=wtl(xs)
    print(f"  {'v4b vs v4a':26}{geomean(xs):>8.3f}x   {w}/{t}/{l}")
    print()

    print("By regime  (geomean)")
    print(f"{'regime':20}{'n':>3}{'v4a/v3':>9}{'v4a/mp':>9}{'v4a/acc':>9}   "
          f"{'v4b/v3':>9}{'v4b/mp':>9}{'v4b/acc':>9}{'v4b/v4a':>9}")
    for g in ("narrow N (<=256)","narrow M (<=128)","all three large","mixed"):
        rs=[r for r in rows if r["regime"]==g]
        if not rs: continue
        print(f"{g:20}{len(rs):>3}"
              f"{geomean([r['v4a_v3'] for r in rs]):>8.3f}x{geomean([r['v4a_mp'] for r in rs]):>8.3f}x"
              f"{geomean([r['v4a_acc'] for r in rs]):>8.3f}x   "
              f"{geomean([r['v4b_v3'] for r in rs]):>8.3f}x{geomean([r['v4b_mp'] for r in rs]):>8.3f}x"
              f"{geomean([r['v4b_acc'] for r in rs]):>8.3f}x"
              f"{geomean([r['v4b_gf']/r['v4a_gf'] for r in rs]):>8.3f}x")
    print()

    print("Where the second profile (v4b) differs from v4a by more than 5%")
    diff=sorted(rows, key=lambda r: -(r["v4b_gf"]/r["v4a_gf"]))
    for r in diff:
        ratio=r["v4b_gf"]/r["v4a_gf"]
        if abs(ratio-1) > TH:
            sign="v4b better" if ratio>1 else "v4a better"
            print(f"  {r['tag']:12}{r['M']}x{r['K']}x{r['N']:<7} v4b/v4a {ratio:.3f}x  {sign}"
                  f"   (spreads {r['v4a_sp']:.1f}% / {r['v4b_sp']:.1f}%)")
    print()
    noisy=sorted(rows, key=lambda r: -max(r['v3_sp'],r['v4a_sp'],r['v4b_sp'],r['mp_sp'],r['acc_sp']))[:5]
    print("Largest spreads (rankings on these shapes are indicative):")
    for r in noisy:
        print(f"  {r['tag']:12}{r['M']}x{r['K']}x{r['N']:<7} max spread "
              f"{max(r['v3_sp'],r['v4a_sp'],r['v4b_sp'],r['mp_sp'],r['acc_sp']):.1f}%  reps={int(r['reps'])}")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "bench/results/2026-09-13/final5.csv")
