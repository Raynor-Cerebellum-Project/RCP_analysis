"""
Time-frequency representation helpers for LFP analysis.

This module contains wavelet, adaptive Morlet, STFT, multitaper, and Paul
wavelet utilities used by plot_lfp_cleaner.py and plotting helpers.

Dependency direction:
    lfp_tfr.py imports lfp_config.py
"""

import numpy as np
from scipy import signal

from RCP_analysis.python.functions.lfp_config import TORCH_AVAILABLE, torch


# =============================================================================
# FREQUENCY / NOTCH HELPERS
# =============================================================================

def get_foi_with_notch_gaps(freq_min, freq_max, n_freqs, notch_freqs=None, notch_width=2,):
    """
    Generate frequencies of interest while avoiding notch regions.

    Parameters
    ----------
    freq_min : float
        Minimum frequency.
    freq_max : float
        Maximum frequency.
    n_freqs : int
        Number of candidate frequencies before notch exclusion.
    notch_freqs : list[float] or None
        Center frequencies to exclude.
    notch_width : float
        Half-width around each notch frequency to exclude.

    Returns
    -------
    foi : ndarray
        Frequencies of interest excluding notch gaps.
    """
    foi = np.linspace(freq_min, freq_max, n_freqs)

    if notch_freqs is None:
        return foi

    keep = np.ones_like(foi, dtype=bool)

    for nf in notch_freqs:
        keep &= ~((foi >= nf - notch_width) & (foi <= nf + notch_width))

    return foi[keep]

def mask_notch_regions(freqs, Sxx, notch_freqs=None, notch_width=2):
    """
    Mask notch frequency regions in an existing spectrogram.

    Returns copies of freqs and Sxx with notch rows removed.
    """
    freqs = np.asarray(freqs)

    if notch_freqs is None:
        return freqs, Sxx

    keep = np.ones_like(freqs, dtype=bool)

    for nf in notch_freqs:
        keep &= ~((freqs >= nf - notch_width) & (freqs <= nf + notch_width))

    return freqs[keep], Sxx[keep, ...]


# =============================================================================
# NORMALIZATION
# =============================================================================

def zscore_normalize_spectrogram(Sxx, times, baseline_window=(-0.5, -0.1), eps=1e-10,):
    """
    Z-score normalize each frequency row using a baseline time window.
    """
    Sxx = np.asarray(Sxx, dtype=float)
    times = np.asarray(times)

    baseline_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])

    if not np.any(baseline_mask):
        return Sxx

    baseline = Sxx[:, baseline_mask]

    mean_val = np.nanmean(baseline, axis=1, keepdims=True)
    std_val = np.nanstd(baseline, axis=1, keepdims=True)

    std_val = np.where(std_val < eps, eps, std_val)

    return (Sxx - mean_val) / std_val


# =============================================================================
# SMALL HELPERS
# =============================================================================

def _next_power_of_two(n): return 1 << (int(n) - 1).bit_length()

def interpolate_nans_1d(x):
    """
    Interpolate NaNs in a one-dimensional array.

    If all values are NaN, returns zeros of same shape.
    """
    x = np.asarray(x, dtype=float)

    if not np.any(np.isnan(x)):
        return x

    good = np.isfinite(x)

    if not np.any(good):
        return np.zeros_like(x)

    idx = np.arange(x.size)
    x_interp = x.copy()
    x_interp[~good] = np.interp(idx[~good], idx[good], x[good])

    return x_interp


# =============================================================================
# MORLET WAVELET METHODS
# =============================================================================

def compute_wavelet_spectrogram(trace, fs, freqs, n_cycles=5,):
    """
    Compute a Morlet wavelet spectrogram for one trace using scipy convolution.

    Returns
    -------
    power : ndarray
        Shape: n_freqs x n_times
    """
    trace = interpolate_nans_1d(trace)
    trace = np.asarray(trace, dtype=float)

    n_times = trace.size
    power = np.full((len(freqs), n_times), np.nan, dtype=float)

    for i, freq in enumerate(freqs):
        sigma_t = n_cycles / (2 * np.pi * freq)
        half_width = int(np.ceil(4 * sigma_t * fs))

        t_wave = np.arange(-half_width, half_width + 1) / fs

        wavelet = np.exp(2j * np.pi * freq * t_wave) * np.exp(
            -(t_wave ** 2) / (2 * sigma_t ** 2)
        )

        wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))

        conv = signal.fftconvolve(trace, wavelet, mode="same")
        power[i, :] = np.abs(conv) ** 2

    return power


def compute_wavelet_spectrogram_batch_torch(traces, fs, freqs, n_cycles=5, device=None, batch_size=64):
    """
    Compute Morlet wavelet power for many traces using torch if available.

    Parameters
    ----------
    traces : ndarray
        Shape: n_traces x n_times
    fs : float
        Sampling frequency.
    freqs : ndarray
        Frequencies of interest.
    n_cycles : float
        Morlet cycles.
    device : str or torch.device or None
        If None, uses cuda if available, else cpu.
    batch_size : int
        Number of traces per batch.

    Returns
    -------
    power : ndarray
        Shape: n_traces x n_freqs x n_times
    """
    traces = np.asarray(traces, dtype=np.float32)
    freqs = np.asarray(freqs, dtype=np.float32)

    if traces.ndim == 1:
        traces = traces[None, :]

    n_traces, n_times = traces.shape

    if not TORCH_AVAILABLE:
        out = np.empty((n_traces, len(freqs), n_times), dtype=np.float32)
        for i in range(n_traces):
            out[i] = compute_wavelet_spectrogram(
                traces[i],
                fs,
                freqs,
                n_cycles=n_cycles,
            ).astype(np.float32)
        return out

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output = np.empty((n_traces, len(freqs), n_times), dtype=np.float32)

    for start in range(0, n_traces, batch_size):
        stop = min(start + batch_size, n_traces)

        batch_np = traces[start:stop]
        batch_np = np.apply_along_axis(interpolate_nans_1d, 1, batch_np).astype(np.float32)

        batch = torch.as_tensor(batch_np, dtype=torch.float32, device=device)

        batch_power_freqs = []

        for freq in freqs:
            sigma_t = n_cycles / (2 * np.pi * float(freq))
            half_width = int(np.ceil(4 * sigma_t * fs))

            t_wave = torch.arange(
                -half_width,
                half_width + 1,
                device=device,
                dtype=torch.float32,
            ) / fs

            wavelet_real = torch.cos(2 * np.pi * float(freq) * t_wave) * torch.exp(
                -(t_wave ** 2) / (2 * sigma_t ** 2)
            )
            wavelet_imag = torch.sin(2 * np.pi * float(freq) * t_wave) * torch.exp(
                -(t_wave ** 2) / (2 * sigma_t ** 2)
            )

            norm = torch.sqrt(torch.sum(wavelet_real ** 2 + wavelet_imag ** 2))
            wavelet_real = wavelet_real / norm
            wavelet_imag = wavelet_imag / norm

            pad = half_width

            x = batch[:, None, :]

            wr = wavelet_real.flip(0)[None, None, :]
            wi = wavelet_imag.flip(0)[None, None, :]

            conv_real = torch.nn.functional.conv1d(x, wr, padding=pad)
            conv_imag = torch.nn.functional.conv1d(x, wi, padding=pad)

            conv_real = conv_real[:, 0, :n_times]
            conv_imag = conv_imag[:, 0, :n_times]

            p = conv_real ** 2 + conv_imag ** 2
            batch_power_freqs.append(p)

        batch_power = torch.stack(batch_power_freqs, dim=1)

        output[start:stop] = batch_power.detach().cpu().numpy()

        del batch
        del batch_power
        del batch_power_freqs

        if device == "cuda":
            torch.cuda.empty_cache()

    return output


def compute_wavelet_spectrogram_batch_torch_adaptive(traces,fs,freqs,adaptive_cycles_low=3,adaptive_cycles_high=7,
    adaptive_freq_low_max=15,adaptive_freq_high_min=16,device=None,batch_size=64):
    """
    Compute adaptive-cycle Morlet wavelet power for many traces using torch.

    Low frequencies use adaptive_cycles_low.
    High frequencies use adaptive_cycles_high.
    Frequencies between adaptive_freq_low_max and adaptive_freq_high_min are
    linearly interpolated.
    """
    traces = np.asarray(traces, dtype=np.float32)
    freqs = np.asarray(freqs, dtype=np.float32)

    if traces.ndim == 1:
        traces = traces[None, :]

    n_traces, n_times = traces.shape

    if not TORCH_AVAILABLE:
        out = np.empty((n_traces, len(freqs), n_times), dtype=np.float32)
        for i in range(n_traces):
            out[i] = compute_morlet_adaptive_spectrogram(
                traces[i],
                fs,
                freqs,
                adaptive_cycles_low=adaptive_cycles_low,
                adaptive_cycles_high=adaptive_cycles_high,
                adaptive_freq_low_max=adaptive_freq_low_max,
                adaptive_freq_high_min=adaptive_freq_high_min,
            ).astype(np.float32)
        return out

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output = np.empty((n_traces, len(freqs), n_times), dtype=np.float32)

    for start in range(0, n_traces, batch_size):
        stop = min(start + batch_size, n_traces)

        batch_np = traces[start:stop]
        batch_np = np.apply_along_axis(interpolate_nans_1d, 1, batch_np).astype(np.float32)

        batch = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
        x = batch[:, None, :]

        batch_power_freqs = []

        for freq in freqs:
            freq_float = float(freq)

            if freq_float <= adaptive_freq_low_max:
                n_cycles = adaptive_cycles_low
            elif freq_float >= adaptive_freq_high_min:
                n_cycles = adaptive_cycles_high
            else:
                frac = (
                    (freq_float - adaptive_freq_low_max)
                    / (adaptive_freq_high_min - adaptive_freq_low_max)
                )
                n_cycles = (
                    adaptive_cycles_low
                    + frac * (adaptive_cycles_high - adaptive_cycles_low)
                )

            sigma_t = n_cycles / (2 * np.pi * freq_float)
            half_width = int(np.ceil(4 * sigma_t * fs))

            t_wave = torch.arange(
                -half_width,
                half_width + 1,
                device=device,
                dtype=torch.float32,
            ) / fs

            gaussian = torch.exp(-(t_wave ** 2) / (2 * sigma_t ** 2))

            wavelet_real = torch.cos(2 * np.pi * freq_float * t_wave) * gaussian
            wavelet_imag = torch.sin(2 * np.pi * freq_float * t_wave) * gaussian

            norm = torch.sqrt(torch.sum(wavelet_real ** 2 + wavelet_imag ** 2))
            wavelet_real = wavelet_real / norm
            wavelet_imag = wavelet_imag / norm

            pad = half_width

            wr = wavelet_real.flip(0)[None, None, :]
            wi = wavelet_imag.flip(0)[None, None, :]

            conv_real = torch.nn.functional.conv1d(x, wr, padding=pad)
            conv_imag = torch.nn.functional.conv1d(x, wi, padding=pad)

            conv_real = conv_real[:, 0, :n_times]
            conv_imag = conv_imag[:, 0, :n_times]

            p = conv_real ** 2 + conv_imag ** 2
            batch_power_freqs.append(p)

        batch_power = torch.stack(batch_power_freqs, dim=1)

        output[start:stop] = batch_power.detach().cpu().numpy()

        del batch
        del x
        del batch_power
        del batch_power_freqs

        if device == "cuda":
            torch.cuda.empty_cache()

    return output


def compute_morlet_adaptive_spectrogram(trace,fs,freqs,adaptive_cycles_low=3,adaptive_cycles_high=7,
    adaptive_freq_low_max=15,adaptive_freq_high_min=16,):
    """
    Compute adaptive-cycle Morlet wavelet spectrogram for one trace.
    """
    trace = interpolate_nans_1d(trace)
    trace = np.asarray(trace, dtype=float)

    n_times = trace.size
    power = np.full((len(freqs), n_times), np.nan, dtype=float)

    for i, freq in enumerate(freqs):
        if freq <= adaptive_freq_low_max:
            n_cycles = adaptive_cycles_low
        elif freq >= adaptive_freq_high_min:
            n_cycles = adaptive_cycles_high
        else:
            frac = (
                (freq - adaptive_freq_low_max)
                / (adaptive_freq_high_min - adaptive_freq_low_max)
            )
            n_cycles = adaptive_cycles_low + frac * (
                adaptive_cycles_high - adaptive_cycles_low
            )

        sigma_t = n_cycles / (2 * np.pi * freq)
        half_width = int(np.ceil(4 * sigma_t * fs))

        t_wave = np.arange(-half_width, half_width + 1) / fs

        wavelet = np.exp(2j * np.pi * freq * t_wave) * np.exp(
            -(t_wave ** 2) / (2 * sigma_t ** 2)
        )

        wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))

        conv = signal.fftconvolve(trace, wavelet, mode="same")
        power[i, :] = np.abs(conv) ** 2

    return power


# =============================================================================
# STFT
# =============================================================================

def compute_stft_spectrogram(trace,fs,freq_min=1,freq_max=120,window_ms=250,step_ms=10):
    """
    Compute STFT spectrogram for one trace.
    """
    trace = interpolate_nans_1d(trace)

    nperseg = int(round(window_ms * fs / 1000))
    step_samples = int(round(step_ms * fs / 1000))
    noverlap = max(0, nperseg - step_samples)

    freqs, times, Sxx = signal.spectrogram(
        trace,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        mode="psd",
    )

    keep = (freqs >= freq_min) & (freqs <= freq_max)

    return freqs[keep], times, Sxx[keep, :]


# =============================================================================
# MULTITAPER
# =============================================================================

def compute_multitaper_spectrogram(trace, fs, freqs, window_ms=500,step_ms=25,time_bandwidth=5,n_tapers=9):
    """
    Compute multitaper spectrogram for one trace.

    This is primarily for diagnostic comparison. Production currently uses
    adaptive Morlet.
    """
    trace = interpolate_nans_1d(trace)
    trace = np.asarray(trace, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    nperseg = int(round(window_ms * fs / 1000))
    step_samples = int(round(step_ms * fs / 1000))

    if nperseg <= 1:
        raise ValueError("window_ms is too small for sampling rate")

    if step_samples <= 0:
        raise ValueError("step_ms is too small for sampling rate")

    if nperseg > trace.size:
        nperseg = trace.size

    starts = np.arange(0, trace.size - nperseg + 1, step_samples, dtype=int)
    times = (starts + nperseg / 2) / fs

    tapers = signal.windows.dpss(
        nperseg,
        NW=time_bandwidth,
        Kmax=n_tapers,
        sym=False,
    )

    Sxx = np.full((len(freqs), len(times)), np.nan, dtype=float)

    n_fft = _next_power_of_two(nperseg)
    fft_freqs = np.fft.rfftfreq(n_fft, d=1 / fs)

    for ti, start in enumerate(starts):
        segment = trace[start:start + nperseg]
        segment = segment - np.nanmean(segment)

        taper_powers = []

        for taper in tapers:
            tapered = segment * taper
            fft_vals = np.fft.rfft(tapered, n=n_fft)
            power = np.abs(fft_vals) ** 2
            taper_powers.append(power)

        mean_power = np.mean(taper_powers, axis=0)

        Sxx[:, ti] = np.interp(freqs, fft_freqs, mean_power)

    return freqs, times, Sxx


# =============================================================================
# PAUL WAVELET
# =============================================================================

def compute_paul_wavelet_spectrogram(trace,fs,freqs,order=4):
    """
    Compute a Paul wavelet spectrogram for one trace.

    This is included for diagnostic comparison, not current production.
    """
    trace = interpolate_nans_1d(trace)
    trace = np.asarray(trace, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    n_times = trace.size
    power = np.full((len(freqs), n_times), np.nan, dtype=float)

    for i, freq in enumerate(freqs):
        # Approximate scale-frequency relationship for Paul wavelet.
        scale = (2 * order + 1) / (4 * np.pi * freq)
        half_width = int(np.ceil(8 * scale * fs))

        t_wave = np.arange(-half_width, half_width + 1) / fs
        x = t_wave / scale

        wavelet = (1 - 1j * x) ** (-(order + 1))
        wavelet *= np.exp(2j * np.pi * freq * t_wave)

        norm = np.sqrt(np.sum(np.abs(wavelet) ** 2))
        if norm > 0:
            wavelet = wavelet / norm

        conv = signal.fftconvolve(trace, wavelet, mode="same")
        power[i, :] = np.abs(conv) ** 2

    return power


# =============================================================================
# DISPATCHER
# =============================================================================

def compute_tfr_method(trace,fs,foi,method_config,times=None):
    """
    Compute one TFR method for a single trace.

    Parameters
    ----------
    trace : ndarray
        One-dimensional LFP trace.
    fs : float
        Sampling frequency.
    foi : ndarray
        Frequencies of interest.
    method_config : dict
        Method-specific config. Must include key "method".
    times : ndarray or None
        Optional time axis for z-score normalization.

    Returns
    -------
    freqs : ndarray
    tfr_times : ndarray
    Sxx : ndarray
    """
    method = method_config.get("method", "morlet")

    if method == "morlet":
        Sxx = compute_wavelet_spectrogram(
            trace,
            fs,
            foi,
            n_cycles=method_config.get("n_cycles", 5),
        )
        tfr_times = times if times is not None else np.arange(len(trace)) / fs
        freqs = foi

    elif method == "morlet_adaptive":
        Sxx = compute_morlet_adaptive_spectrogram(
            trace,
            fs,
            foi,
            adaptive_cycles_low=method_config.get("adaptive_cycles_low", 3),
            adaptive_cycles_high=method_config.get("adaptive_cycles_high", 7),
            adaptive_freq_low_max=method_config.get("adaptive_freq_low_max", 15),
            adaptive_freq_high_min=method_config.get("adaptive_freq_high_min", 16),
        )
        tfr_times = times if times is not None else np.arange(len(trace)) / fs
        freqs = foi

    elif method == "stft":
        freq_min = float(np.min(foi))
        freq_max = float(np.max(foi))

        freqs, stft_times, Sxx = compute_stft_spectrogram(
            trace,
            fs,
            freq_min=freq_min,
            freq_max=freq_max,
            window_ms=method_config.get("window_ms", 250),
            step_ms=method_config.get("step_ms", 10),
        )

        if times is not None:
            tfr_times = stft_times + times[0]
        else:
            tfr_times = stft_times

    elif method == "multitaper":
        freqs, mt_times, Sxx = compute_multitaper_spectrogram(
            trace,
            fs,
            foi,
            window_ms=method_config.get("window_ms", 500),
            step_ms=method_config.get("step_ms", 25),
            time_bandwidth=method_config.get("time_bandwidth", 5),
            n_tapers=method_config.get("n_tapers", 9),
        )

        if times is not None:
            tfr_times = mt_times + times[0]
        else:
            tfr_times = mt_times

    elif method == "paul":
        Sxx = compute_paul_wavelet_spectrogram(
            trace,
            fs,
            foi,
            order=method_config.get("order", 4),
        )
        tfr_times = times if times is not None else np.arange(len(trace)) / fs
        freqs = foi

    else:
        raise ValueError(f"Unknown TFR method: {method}")

    normalize = method_config.get("normalize", None)

    if normalize == "zscore":
        baseline_window = method_config.get("baseline_window", (-0.5, -0.1))
        Sxx = zscore_normalize_spectrogram(
            Sxx,
            tfr_times,
            baseline_window=baseline_window,
        )

    return freqs, tfr_times, Sxx