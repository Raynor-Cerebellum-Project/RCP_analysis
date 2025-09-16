#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")   # ensure headless
import matplotlib.pyplot as plt

def save_rate_plots(npz_path: Path, out_dir: Path, channels=None, tlim=None):
    """
    Load saved firing rate file and save plots to disk.
    """
    data = np.load(npz_path, allow_pickle=True)
    rate_hz = data["rate_hz"]          # (n_channels, n_bins)
    t_ms    = data["t_ms"]             # (n_bins,)
    meta    = data.get("meta", None)

    n_ch, n_bins = rate_hz.shape
    if channels is None:
        channels = list(range(min(5, n_ch)))  # default: first 5 channels

    # Restrict to time window if provided
    if tlim is not None:
        mask = (t_ms >= tlim[0]) & (t_ms <= tlim[1])
        t_ms = t_ms[mask]
        rate_hz = rate_hz[:, mask]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Use meta info in titles/filenames if available
    if meta is not None:
        bin_ms = meta.item().get("bin_ms", None) if hasattr(meta, "item") else meta.get("bin_ms", None)
        sigma_ms = meta.item().get("sigma_ms", None) if hasattr(meta, "item") else meta.get("sigma_ms", None)
        subtitle = f"(bin={bin_ms} ms, sigma={sigma_ms} ms)" if (bin_ms and sigma_ms) else ""
    else:
        subtitle = ""

    # 1) Line plot of selected channels
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

    # 2) Heatmap of all channels
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

if __name__ == "__main__":
    npz_file = Path(
        "/home/bryan/mnt/cullen/Current Project Databases - NHP/2025 Cerebellum prosthesis/"
        "Bryan/RCP_analysis/results/checkpoint/rates__Nike_reaching_002_002__bin1ms_sigma20ms.npz"
    )
    out_dir = Path(
        "/home/bryan/mnt/cullen/Current Project Databases - NHP/2025 Cerebellum prosthesis/"
        "Bryan/RCP_analysis/results/figures"
    )
    save_rate_plots(npz_file, out_dir, channels=[0, 10, 20], tlim=(0, 5000))  # first 5 s
