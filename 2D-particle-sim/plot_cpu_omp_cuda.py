import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(".")

CPU_PATH = ROOT / "cpuSim" / "results_cpu.csv"
OMP8_PATH = ROOT / "cpuSim" / "results_omp_8t.csv"
OMP1_PATH = ROOT / "cpuSim" / "results_omp_1t.csv"  # optional
CUDA_PATH = ROOT / "gpuSim" / "timings_cuda.csv"

def load_cpu_like(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["mode"] == "bench"].copy()
    df["label"] = label
    df["N"] = df["n_target"].astype(int)
    df["ms"] = df["avg_ms_evolve"].astype(float)
    return df[["label", "N", "ms"]]

def load_cuda_step(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = label
    df["N"] = df["N"].astype(int)
    df["ms"] = df["ms_per_step"].astype(float)
    return df[["label", "N", "ms"]]

dfs = []

if CPU_PATH.exists():
    dfs.append(load_cpu_like(CPU_PATH, "CPU"))
else:
    raise RuntimeError(f"Missing {CPU_PATH}")

if OMP8_PATH.exists():
    dfs.append(load_cpu_like(OMP8_PATH, "OMP 8T"))
else:
    print(f"Warning: missing {OMP8_PATH}, skipping OMP 8T")

if OMP1_PATH.exists():
    dfs.append(load_cpu_like(OMP1_PATH, "OMP 1T"))

if CUDA_PATH.exists():
    dfs.append(load_cuda_step(CUDA_PATH, "CUDA"))
else:
    raise RuntimeError(f"Missing {CUDA_PATH}")

all_df = pd.concat(dfs, ignore_index=True)
all_df = all_df.sort_values(["label", "N"])

# Plot 1: ms vs N (log-log)
plt.figure()
for label, g in all_df.groupby("label"):
    g = g.sort_values("N")
    plt.plot(g["N"], g["ms"], marker="o", label=label)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Particles (N)")
plt.ylabel("Milliseconds (ms)")
plt.title("CPU vs OMP vs CUDA (time per evolve/step)")
plt.legend()
plt.tight_layout()
plt.savefig("plot_cpu_omp_cuda_ms.png", dpi=200)

# Plot 2: speedup vs CPU (based on common N values)
cpu_ms = all_df[all_df["label"] == "CPU"].set_index("N")["ms"]

plt.figure()
for label in [x for x in all_df["label"].unique() if x != "CPU"]:
    ms = all_df[all_df["label"] == label].set_index("N")["ms"]
    common = cpu_ms.index.intersection(ms.index)
    if len(common) == 0:
        continue
    speedup = (cpu_ms.loc[common] / ms.loc[common]).sort_index()
    plt.plot(speedup.index, speedup.values, marker="o", label=f"{label} speedup")

plt.xscale("log")
plt.xlabel("Particles (N)")
plt.ylabel("Speedup vs CPU (x)")
plt.title("Speedup vs CPU")
plt.legend()
plt.tight_layout()
plt.savefig("plot_cpu_omp_cuda_speedup.png", dpi=200)

print("Wrote plot_cpu_omp_cuda_ms.png and plot_cpu_omp_cuda_speedup.png")
