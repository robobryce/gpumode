"""brief-46 COMBINE interleaved A/B driver.

Runs N rounds of A then B as FRESH subprocesses (so the QR_2L_PNW constexpr is
re-read each arm), all inside the SINGLE exclusive GPU lock the harness wrapper
holds. Parses each arm's `RESULT pnw=.. geomean=.. n4096=..` tail and reports
per-round + aggregate medians so the NET geomean and the isolated-n4096 delta
can be judged round-by-round (robust to remote infra variance per memory).

  ARM A = parent  : QR_2L_PNW=0  (n4096 tall sub-panel nwp=32, in-code default)
  ARM B = candidate: QR_2L_PNW=8 (n4096 tall sub-panel nwp=8, graft default)
"""
import os, sys, subprocess, statistics, re

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def run_arm(pnw):
    env = dict(os.environ)
    env["QR_2L_PNW"] = str(pnw)
    p = subprocess.run([PY, os.path.join(HERE, "_brief46_ab_one.py")],
                       cwd=HERE, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout + "\n" + p.stderr + "\n")
        raise SystemExit(f"arm pnw={pnw} failed rc={p.returncode}")
    m = re.search(r"RESULT pnw=\S+ geomean=([\d.]+) n4096=([\d.]+)", p.stdout)
    if not m:
        sys.stderr.write(p.stdout + "\n")
        raise SystemExit(f"arm pnw={pnw}: no RESULT line")
    return float(m.group(1)), float(m.group(2)), p.stdout


A_geo, A_n4, B_geo, B_n4 = [], [], [], []
for r in range(ROUNDS):
    ag, an, aout = run_arm(0)   # parent
    bg, bn, bout = run_arm(8)   # candidate
    A_geo.append(ag); A_n4.append(an)
    B_geo.append(bg); B_n4.append(bn)
    print(f"--- round {r}: A(parent) geo={ag:8.2f} n4096={an:8.2f} | "
          f"B(cand) geo={bg:8.2f} n4096={bn:8.2f} | "
          f"d_geo={(bg-ag)/ag*100:+6.2f}%  d_n4096={(bn-an)/an*100:+6.2f}%")
    sys.stdout.flush()


def med(x):
    return statistics.median(x)


print("=" * 70)
print(f"AGGREGATE over {ROUNDS} rounds (medians):")
print(f"  A parent   : geomean={med(A_geo):8.2f}us  n4096={med(A_n4):8.2f}us")
print(f"  B candidate: geomean={med(B_geo):8.2f}us  n4096={med(B_n4):8.2f}us")
dg = (med(B_geo) - med(A_geo)) / med(A_geo) * 100
dn = (med(B_n4) - med(A_n4)) / med(A_n4) * 100
print(f"  DELTA geomean = {dg:+.2f}%   DELTA n4096 = {dn:+.2f}%")
print(f"  candidate {'WINS' if med(B_geo) < med(A_geo) else 'LOSES/TIES'} on NET geomean")
