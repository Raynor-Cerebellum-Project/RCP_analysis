from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from RCP_analysis import load_experiment_params

""" This script plots the first 5 seconds of the Blackrock file, for plotting both UA/BR and NPRW/Intan use NPRW_UA_FR_plotting"""



def resolve_output_root(p) -> Path:
    base = Path(p.data_root).resolve()
    out = getattr(p, "output_root", None)
    if out:
        return Path(out).resolve() if str(out).startswith("/") else (base / out).resolve()
    # fallback if output_root not set in YAML
    return (base / "results").resolve()

def save_rate_plots(npz_path: Path, out_dir: Path, channels=None, tlim=None):
    data = np.load(npz_path, allow_pickle=True)
    rate_hz = data["rate_hz"]
    t_ms    = data["t_ms"]
    meta    = data.get("meta", None)
    pcs = data.get("pcs", None)
    explained_var = data.get("explained_var", None)

    n_ch, _ = rate_hz.shape
    if channels is None:
        channels = list(range(min(5, n_ch)))

    if tlim is not None:
        mask = (t_ms >= tlim[0]) & (t_ms <= tlim[1])
        t_ms = t_ms[mask]
        rate_hz = rate_hz[:, mask]
        if pcs is not None:
            pcs = pcs[:, mask]

    out_dir.mkdir(parents=True, exist_ok=True)

    if meta is not None:
        meta_dict = meta.item() if hasattr(meta, "item") else meta
        bin_ms = meta_dict.get("bin_ms", None)
        sigma_ms = meta_dict.get("sigma_ms", None)
        subtitle = f"(bin={bin_ms} ms, sigma={sigma_ms} ms)" if (bin_ms and sigma_ms) else ""
    else:
        subtitle = ""

    # 1) line plot
    plt.figure(figsize=(10, 6))
    for ch in channels:
        plt.plot(t_ms/1000, rate_hz[ch], label=f"Ch {ch}")
    plt.xlabel("Time (s)")
    plt.ylabel("Firing rate (Hz)")
    plt.title(f"Firing rates: {npz_path.stem} {subtitle}")
    plt.legend()
    plt.tight_layout()
    out_line = out_dir / f"{npz_path.stem}_line.png"
    plt.savefig(out_line, dpi=150)
    plt.close()
    print(f"Saved line plot -> {out_line}")

    # 2) heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(rate_hz, aspect="auto", cmap="hot",
               extent=[t_ms[0]/1000, t_ms[-1]/1000, n_ch, 0])
    plt.colorbar(label="Firing rate (Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Channel index")
    plt.title(f"Firing rate heatmap: {npz_path.stem} {subtitle}")
    plt.tight_layout()
    out_heat = out_dir / f"{npz_path.stem}_heatmap.png"
    plt.savefig(out_heat, dpi=150)
    plt.close()
    print(f"Saved heatmap -> {out_heat}")

    # 3) PCA
    if pcs is not None:
        plt.figure(figsize=(10, 6))
        n_pcs = pcs.shape[0]
        for i in range(n_pcs):
            var_txt = f" ({explained_var[i]*100:.1f}%)" if explained_var is not None else ""
            plt.plot(t_ms/1000, pcs[i], label=f"PC{i+1}{var_txt}")
        plt.xlabel("Time (s)")
        plt.ylabel("PC projection (a.u.)")
        plt.title(f"PCA trajectories: {npz_path.stem} {subtitle}")
        plt.legend()
        plt.tight_layout()
        out_pcs = out_dir / f"{npz_path.stem}_pcs.png"
        plt.savefig(out_pcs, dpi=150)
        plt.close()
        print(f"Saved PCA plot -> {out_pcs}")

if __name__ == "__main__":
    # Load YAML
    REPO_ROOT = Path(__file__).resolve().parents[1]
    PARAMS = load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)

    # Resolve output root from YAML (absolute or relative to data_root)
    OUT_ROOT = resolve_output_root(PARAMS)
    CHECKPOINT_DIR = OUT_ROOT / "checkpoint"
    FIG_DIR = OUT_ROOT / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Output root: {OUT_ROOT}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Figures dir: {FIG_DIR}")

    # Loop through all .npz files that match rates__*__.npz
    for npz_file in sorted(CHECKPOINT_DIR.glob("rates__*.npz")):
        print(f"Processing {npz_file.name}")
        save_rate_plots(npz_file, FIG_DIR, channels=[0, 10, 20], tlim=(0, 5000))