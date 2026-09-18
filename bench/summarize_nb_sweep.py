#!/usr/bin/env python3
"""Summarise bench_nb_sweep output.

Two different "best" answers, reported separately because they answer different
questions:
  A  one fixed blocking for all 35 shapes, ranked by the geometric mean of its
     speed ratio against the fixed internal reference Mc16/Nc256/Kc2048
  B  each shape's own best blockings

B is a table of measured per-shape leaders. It is NOT the result of an
implemented dispatch and must not be read as an achievable single-kernel number.

Candidates whose result did not match the v3 reference are dropped everywhere.
"""
import csv, sys, math, collections

TIE = 0.03   # within 3% of a shape's leader counts as a tie for that shape
TH  = 0.05   # repo-wide "is this real" threshold

def geomean(xs):
    xs=[x for x in xs if x and x>0]
    return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else float("nan")

def regime(M,K,N):
    if N<=256: return "narrow N (<=256)"
    if M<=128: return "narrow M (<=128)"
    if M>=1024 and K>=1024 and N>=1024: return "all three large"
    return "mixed"

def load(path, stage):
    rows=[]
    for r in csv.DictReader(open(path)):
        if r["stage"]!=stage: continue
        if r["correct"]!="ok": continue
        rows.append(dict(key=(r["tag"],int(r["M"]),int(r["K"]),int(r["N"])),
                         tag=r["tag"], M=int(r["M"]), K=int(r["K"]), N=int(r["N"]),
                         Mc=int(r["Mc"]), Nc=int(r["Nc"]), Kc=int(r["Kc"]),
                         ms=float(r["ms"]), gf=float(r["gflops"]),
                         sp=float(r["spread"]), reps=int(r["reps"]),
                         packA=float(r["packA_MiB"]), packB=float(r["packB_MiB"]),
                         apk=int(r["a_packs"]), bpk=int(r["b_packs"]),
                         vs_ref=float(r["vs_ref"]) if r["vs_ref"] else None))
    return rows

def main(path):
    pre  = load(path,"prescan")
    fin  = load(path,"final")
    bad  = sum(1 for r in csv.DictReader(open(path)) if r["correct"]!="ok")
    shapes = sorted({r["key"] for r in pre}, key=lambda k:(k[0],k[1],k[2],k[3]))
    print(f"prescan rows {len(pre)}, final rows {len(fin)}, shapes {len(shapes)}, "
          f"incorrect candidates dropped {bad}\n")

    # ---------- A: one fixed blocking for everything ----------
    per = collections.defaultdict(dict)     # combo -> shape -> ratio
    for r in pre:
        if r["vs_ref"]: per[(r["Mc"],r["Nc"],r["Kc"])][r["key"]] = r["vs_ref"]
    full = {c:v for c,v in per.items() if len(v)==len(shapes)}
    rank = sorted(full.items(), key=lambda kv: -geomean(list(kv[1].values())))
    print("A. Best SINGLE fixed blocking over all 35 shapes")
    print("   ratios are vs the fixed internal reference Mc16/Nc256/Kc2048 (prescan)")
    print(f"   {'Mc/Nc/Kc':>16}{'geomean':>9}{'worst':>8}{'best':>8}   >+5%  within  <-5%")
    for c,v in rank[:12]:
        xs=list(v.values())
        w=sum(1 for x in xs if x>1+TH); l=sum(1 for x in xs if x<1-TH)
        print(f"   {f'{c[0]}/{c[1]}/{c[2]}':>16}{geomean(xs):>8.3f}x{min(xs):>7.3f}x{max(xs):>7.3f}x"
              f"{w:>7}{len(xs)-w-l:>8}{l:>6}")
    print("   ... worst three:")
    for c,v in rank[-3:]:
        xs=list(v.values())
        print(f"   {f'{c[0]}/{c[1]}/{c[2]}':>16}{geomean(xs):>8.3f}x{min(xs):>7.3f}x{max(xs):>7.3f}x")
    print()

    # ---------- B: per-shape leaders, from the longer stage ----------
    print("B. Per-shape measured leaders (stage 2). NOT an implemented dispatch.")
    byshape=collections.defaultdict(list)
    for r in fin: byshape[r["key"]].append(r)
    for key in shapes:
        rs=sorted(byshape.get(key,[]), key=lambda r: r["ms"])
        if not rs: continue
        tag,M,K,N=key
        best=rs[0]["ms"]
        line=[]
        for r in rs[:3]:
            tie=" =" if r["ms"]<=best*(1+TIE) and r is not rs[0] else ""
            line.append(f"{r['Mc']}/{r['Nc']}/{r['Kc']} {r['gf']:.0f}GF ±{r['sp']:.1f}%{tie}")
        ties=sum(1 for r in rs if r["ms"]<=best*(1+TIE))
        print(f"  {tag:12}{M}x{K}x{N:<7} " + " | ".join(line) + f"   ({ties} within {int(TIE*100)}%)")
    print()

    # ---------- trends ----------
    print("Trends: geomean ratio vs reference, prescan, one axis at a time")
    for axis in ("Mc","Nc","Kc"):
        print(f"  {axis}:", end="")
        vals=sorted({r[axis] for r in pre})
        for v in vals:
            xs=[r["vs_ref"] for r in pre if r[axis]==v and r["vs_ref"]]
            print(f"  {v}:{geomean(xs):.3f}x", end="")
        print()
    print()
    print("Per regime, best fixed blocking within that regime (prescan geomean)")
    for g in ("narrow N (<=256)","narrow M (<=128)","all three large","mixed"):
        tags={r["key"] for r in pre if regime(r["M"],r["K"],r["N"])==g}
        if not tags: continue
        sub=collections.defaultdict(list)
        for r in pre:
            if r["key"] in tags and r["vs_ref"]: sub[(r["Mc"],r["Nc"],r["Kc"])].append(r["vs_ref"])
        top=sorted(sub.items(), key=lambda kv:-geomean(kv[1]))[:3]
        print(f"  {g:20} n={len(tags):2}  " +
              "  ".join(f"{c[0]}/{c[1]}/{c[2]} {geomean(v):.3f}x" for c,v in top))
    print()
    noisy=sorted(fin, key=lambda r:-r["sp"])[:5]
    print("Largest stage-2 spreads (treat these shapes' rankings as indicative):")
    for r in noisy:
        print(f"  {r['tag']:12}{r['M']}x{r['K']}x{r['N']:<7} {r['Mc']}/{r['Nc']}/{r['Kc']}  ±{r['sp']:.1f}%  reps={r['reps']}")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "bench/results/2026-09-13/nb_sweep.csv")
