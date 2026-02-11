import pandas as pd
from pathlib import Path

ROOT = Path(".")

CPU = ROOT / "cpuSim" / "results_cpu.csv"
OMP8 = ROOT / "cpuSim" / "results_omp_8t.csv"
OMP1 = ROOT / "cpuSim" / "results_omp_1t.csv"
CUDA = ROOT / "gpuSim" / "timings_cuda.csv"

def load_cpu_like(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["mode"] = df["mode"].astype(str).str.strip()
    df = df[df["mode"] == "bench"].copy()

    N = pd.to_numeric(df["n_target"], errors="coerce")
    ms = pd.to_numeric(df["avg_ms_evolve"], errors="coerce")

    out = pd.DataFrame({
        "backend": [label] * len(df),
        "N": N,
        "ms": ms
    })

    out = out.dropna(subset=["N", "ms"]).copy()
    out["N"] = out["N"].astype(int)
    return out

def load_cuda(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    N = pd.to_numeric(df["N"], errors="coerce")
    ms = pd.to_numeric(df["ms_per_step"], errors="coerce")

    out = pd.DataFrame({
        "backend": [label] * len(df),
        "N": N,
        "ms": ms
    })

    out = out.dropna(subset=["N", "ms"]).copy()
    out["N"] = out["N"].astype(int)
    return out

rows = [
    load_cpu_like(CPU, "CPU"),
    load_cpu_like(OMP1, "OMP 1T"),
    load_cpu_like(OMP8, "OMP 8T"),
    load_cuda(CUDA, "CUDA")
]

df = pd.concat(rows, ignore_index=True)

print("Loaded backends:", df["backend"].unique().tolist())
print("Rows per backend:\n", df.groupby("backend")["N"].count().to_string())

pivot = df.pivot_table(index="N", columns="backend", values="ms", aggfunc="mean").reset_index()

# speedups vs CPU
for col in ["OMP 1T", "OMP 8T", "CUDA"]:
    if col in pivot.columns:
        pivot[f"{col} speedup"] = pivot["CPU"] / pivot[col]

pivot.to_csv("portfolio_speedup_table.csv", index=False)

disp = pivot.copy()
for c in disp.columns:
    if c != "N":
        disp[c] = disp[c].map(lambda x: "" if pd.isna(x) else float(x))


print("Wrote portfolio_speedup_table.csv")
print("Wrote portfolio_speedup_table.md")

# headline at N=20000
target_n = 20000
r = pivot[pivot["N"] == target_n]
if not r.empty and "CUDA speedup" in pivot.columns:
    r = r.iloc[0]
    print(f"Headline at N=20000: CPU {r['CPU']:.2f} ms, CUDA {r['CUDA']:.3f} ms, speedup {r['CUDA speedup']:.1f}x")

# ---------- simple markdown writer (no tabulate needed) ----------
def df_to_md(df):
    cols = df.columns.tolist()
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"]*len(cols)) + " |")

    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if c == "N":
                vals.append(str(int(v)))
            else:
                vals.append("" if (v == "" or v is None) else (f"{float(v):.4g}" if v == v else ""))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)

from pathlib import Path
Path("portfolio_speedup_table.md").write_text(df_to_md(disp))
print("Wrote portfolio_speedup_table.md (no dependencies)")
