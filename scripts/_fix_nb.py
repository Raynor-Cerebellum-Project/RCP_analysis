import nbformat as nbf

nb_path = r'e:\NHP_Cerebellum_Project_2025_RCP_Analysis_Git\RCP_analysis\scripts\debug_IPCA_artifact_correction.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# ── Cell: Single-pulse zoom view ──
nb.cells.append(nbf.v4.new_markdown_cell("## Zoomed Single-Pulse View"))
nb.cells.append(nbf.v4.new_code_cell("""\
# Zoomed single-pulse view - raw, artifact template, corrected
trial_idx = 0   # which trial to show
pulse_in_trial = 0  # which pulse within that trial (0 = first pulse)

trial_pulses = [(idx, ps, pe) for idx, (t, ps, pe) in enumerate(micro_map) if t == trial_idx]

if not trial_pulses:
    print(f"No pulses found for trial {trial_idx}")
else:
    glob_idx, p_start, p_end = trial_pulses[pulse_in_trial]

    raw_snippet  = micro_signal_array[glob_idx, :, 0]
    corr_snippet = micro_corrected[glob_idx, :, 0]
    art_snippet  = raw_snippet - corr_snippet

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    axes[0].plot(micro_time_axis, raw_snippet,  color='k',    lw=1.5, label='Raw')
    axes[1].plot(micro_time_axis, art_snippet,  color='r',    lw=1.5, label='Artifact template')
    axes[2].plot(micro_time_axis, corr_snippet, color='royalblue', lw=1.5, label='Corrected')

    for ax, title in zip(axes, ['Raw', 'Artifact Template', 'Corrected']):
        ax.axvline(0, color='k', ls=':', alpha=0.5, label='Pulse center')
        ax.set_xlabel('Time (ms)')
        ax.set_title(title)
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Amplitude (uV)')
    fig.suptitle(f'Trial {trial_idx+1}, Pulse {pulse_in_trial+1}', fontsize=13)
    plt.tight_layout()
    plt.show()
"""))

# ── Cell: Linear interpolation blanking ──
nb.cells.append(nbf.v4.new_markdown_cell("## Linear Interpolation Blanking"))
nb.cells.append(nbf.v4.new_code_cell("""\
# Linear interpolation blanking across each stimulus pulse
BLANK_PRE_MS  = 0.1   # ms before detected peak to start interpolation
BLANK_POST_MS = 0.2   # ms after  detected peak to end   interpolation
SEARCH_MS     = 0.15  # ms search radius around predicted time to find UA peak
trial_idx     = 0

blank_pre_samp  = int(round(BLANK_PRE_MS  / 1000.0 * fs))
blank_post_samp = int(round(BLANK_POST_MS / 1000.0 * fs))
search_samp     = int(round(SEARCH_MS     / 1000.0 * fs))

raw_trace    = full_raw_array[trial_idx].copy()
interp_trace = raw_trace.copy()

trial_pulses = [(gi, ps, pe) for gi, (t, ps, pe) in enumerate(micro_map) if t == trial_idx]

shifts_ms = []

for glob_idx, p_start, p_end in trial_pulses:
    predicted_center = (p_start + p_end) // 2

    s0 = max(0, predicted_center - search_samp)
    s1 = min(len(raw_trace), predicted_center + search_samp + 1)

    chunk_diff = np.abs(np.diff(raw_trace[s0:s1], prepend=raw_trace[s0]))
    detected_center = s0 + np.argmax(chunk_diff)

    shifts_ms.append((detected_center - predicted_center) / fs * 1000.0)

    b0 = max(0, detected_center - blank_pre_samp)
    b1 = min(len(interp_trace) - 1, detected_center + blank_post_samp)

    v0 = interp_trace[b0]
    v1 = interp_trace[b1]
    n  = b1 - b0
    if n > 0:
        interp_trace[b0:b1+1] = np.linspace(v0, v1, n + 1)

shifts_ms = np.array(shifts_ms)
print(f"Artifact shift vs prediction: mean={np.mean(shifts_ms):.3f} ms, "
      f"std={np.std(shifts_ms):.3f} ms, range=[{np.min(shifts_ms):.3f}, {np.max(shifts_ms):.3f}] ms")

fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

axes[0].plot(time_axis, raw_trace,    color='k',         lw=0.8, label='Raw (broadband)')
axes[0].set_title(f'Trial {trial_idx+1} - Raw signal')
axes[0].set_ylabel('uV')

axes[1].plot(time_axis, interp_trace, color='orange',    lw=0.8, label='Interpolated')
axes[1].set_title(f'After interpolation (blank -{BLANK_PRE_MS}/+{BLANK_POST_MS} ms per pulse)')
axes[1].set_ylabel('uV')
axes[1].set_xlabel('Time (ms)')

for ax in axes:
    ax.axvline(0, color='g', ls='--', lw=0.8)
    for glob_idx, p_start, p_end in trial_pulses:
        pc = (p_start + p_end) // 2
        t0 = time_axis[max(0, pc - blank_pre_samp)]
        t1 = time_axis[min(len(time_axis)-1, pc + blank_post_samp)]
        ax.axvspan(t0, t1, color='orange', alpha=0.15)
    ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.show()
"""))

# ── Cell: HPF on interpolated data ──
nb.cells.append(nbf.v4.new_markdown_cell("## High-Pass Filter on Interpolated Data"))
nb.cells.append(nbf.v4.new_code_cell("""\
from scipy.signal import butter, filtfilt

HPF_CUTOFF_HZ = 300.0

b, a = butter(4, HPF_CUTOFF_HZ / (fs / 2.0), btype='high')
interp_hpf = filtfilt(b, a, interp_trace)

fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

axes[0].plot(time_axis, raw_trace, color='k', lw=0.8, label='Raw (broadband)')
axes[0].set_title(f'Trial {trial_idx+1} - Raw')
axes[0].set_ylabel('uV')

axes[1].plot(time_axis, interp_trace, color='orange', lw=0.8, label='Interpolated (broadband)')
axes[1].set_title('After linear interpolation blanking')
axes[1].set_ylabel('uV')

axes[2].plot(time_axis, interp_hpf, color='royalblue', lw=0.8, label=f'Interp + HPF {HPF_CUTOFF_HZ:.0f} Hz')
axes[2].set_title(f'After HPF {HPF_CUTOFF_HZ:.0f} Hz')
axes[2].set_ylabel('uV')
axes[2].set_xlabel('Time (ms)')

for ax in axes:
    ax.axvline(0, color='g', ls='--', lw=0.8)
    for glob_idx, p_start, p_end in trial_pulses:
        pc = (p_start + p_end) // 2
        t0 = time_axis[max(0, pc - blank_pre_samp)]
        t1 = time_axis[min(len(time_axis)-1, pc + blank_post_samp)]
        ax.axvspan(t0, t1, color='orange', alpha=0.15)
    ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.show()
"""))

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Done! Notebook now has {len(nb.cells)} cells")
