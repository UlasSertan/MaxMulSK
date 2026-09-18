#!/usr/bin/env python3
"""Summarise bench_v4c3: Accelerate / v4c / MpGEMM."""
import csv, sys, math, collections
TH=0.05
def geo(xs): return math.exp(sum(math.log(x) for x in xs)/len(xs))
def reg(M,K,N):
    if N<=256: return "narrow N (<=256)"
    if M<=128: return "narrow M (<=128)"
    if M>=1024 and K>=1024 and N>=1024: return "all three large"
    return "mixed"
def wtl(xs):
    w=sum(1 for x in xs if x>1+TH); l=sum(1 for x in xs if x<1-TH); return w,len(xs)-w-l,l
def main(p):
    rows=list(csv.DictReader(open(p)))
    nonnative=[r for r in rows if r['v4c_path']!='native']
    print(f"{len(rows)} shapes, {len(nonnative)} not run natively by v4c "
          f"(these would need an explicitly labelled fallback; there are none here)\n")
    print("profile chosen by the rule: " + str(dict(collections.Counter(r['v4c_profile'] for r in rows))))
    print(f"largest relative error vs v3: v4c {max(float(r['err_v4c']) for r in rows):.3e}, "
          f"Accelerate {max(float(r['err_acc']) for r in rows):.3e}, "
          f"MpGEMM {max(float(r['err_mp']) for r in rows):.3e}\n")
    for key,name in (('v4c_over_acc','v4c vs Accelerate'),('v4c_over_mp','v4c vs MpGEMM')):
        xs=[float(r[key]) for r in rows]; w,t,l=wtl(xs)
        print(f"  {name:22} geomean {geo(xs):.3f}x   win/tie/loss {w}/{t}/{l}")
    print(f"\n{'regime':20}{'n':>3}{'v4c/acc':>10}{'v4c/mp':>9}   w/t/l acc    w/t/l mp")
    for g in ("narrow N (<=256)","narrow M (<=128)","all three large","mixed"):
        rs=[r for r in rows if reg(int(r['M']),int(r['K']),int(r['N']))==g]
        if not rs: continue
        a=[float(r['v4c_over_acc']) for r in rs]; m=[float(r['v4c_over_mp']) for r in rs]
        wa,ta,la=wtl(a); wm,tm,lm=wtl(m)
        print(f"{g:20}{len(rs):>3}{geo(a):>9.3f}x{geo(m):>8.3f}x   {wa}/{ta}/{la}        {wm}/{tm}/{lm}")
    print("\nlargest losses")
    for key,name in (('v4c_over_acc','vs Accelerate'),('v4c_over_mp','vs MpGEMM')):
        for r in sorted(rows,key=lambda r: float(r[key]))[:5]:
            print(f"  {name:14}{r['tag']:12}{r['M']}x{r['K']}x{r['N']:<7} {float(r[key]):.3f}x")
    print("\nlargest spreads")
    for r in sorted(rows,key=lambda r:-max(float(r['acc_sp']),float(r['v4c_sp']),float(r['mp_sp'])))[:5]:
        print(f"  {r['tag']:12}{r['M']}x{r['K']}x{r['N']:<7} max "
              f"{max(float(r['acc_sp']),float(r['v4c_sp']),float(r['mp_sp'])):.1f}%  reps={r['reps']}")
if __name__=="__main__": main(sys.argv[1] if len(sys.argv)>1 else "bench/results/2026-09-13/v4c3.csv")
