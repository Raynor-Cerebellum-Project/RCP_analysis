function [amplifier_data_cleaned, stim_drift] = template_subtraction(amplifier_data, trigs, chan, params, template_mode, plot_chan)
% TEMPLATE_SUBTRACTION
% Subtracts pulse-by-pulse artifact templates first, then fits and removes
% exponential decay + oscillation across full blocks, and finally applies filtering.

% === Unpack Parameters ===
NSTIM = params.NSTIM;
buffer = params.buffer;
template_leeway = params.template_leeway;
stim_neural_delay = params.stim_neural_delay;
period_avg = params.period_avg;
window_size = params.movmean_window;
pca_k = params.pca_components;
med_filt_range = params.med_filt_range;
gauss_filt_range = params.gauss_filt_range;
fs = 30000;

trigs_beg = trigs(:, 1);
trigs_end = trigs(:, 2) + stim_neural_delay;

% === Identify Trial Groups ===
time_diffs = diff(trigs_beg);
repeat_gap_threshold = 2 * (2 * buffer + 1);
repeat_boundaries = [0; find(time_diffs > repeat_gap_threshold); numel(trigs_beg)];
num_repeats = numel(repeat_boundaries) - 1;
num_pulse = NSTIM / num_repeats;

% === Non-Parametric Drift Removal: Median + Gaussian Filter ===
stim_drift = struct('values', cell(num_repeats, 1), 'indices', cell(num_repeats, 1));

for r = 1:num_repeats
    idx = (repeat_boundaries(r)+1):repeat_boundaries(r+1);
    full_range_start = trigs_end(idx(1))-4;

    if r < num_repeats
        range_end = trigs_beg(repeat_boundaries(r+1)+1) - 1;
    else
        range_end = trigs_end(idx(end)) + buffer;
    end

    if full_range_start > 0 && range_end <= length(amplifier_data)
        raw_block = double(amplifier_data(full_range_start:range_end));
        med_filtered = medfilt1(raw_block, med_filt_range);
        gauss_filtered = imgaussfilt(med_filtered, gauss_filt_range);

        amplifier_data(full_range_start:range_end) = amplifier_data(full_range_start:range_end) - gauss_filtered;
        stim_drift(r).values = gauss_filtered(:);
        stim_drift(r).indices = full_range_start:range_end;

        if r == 1 && chan == plot_chan
            t_ms_1 = (full_range_start:range_end) / fs * 1000;
            raw_block_1 = raw_block;
            med_filtered_1 = med_filtered;
            gauss_filtered_1 = gauss_filtered;
            full_range_start_1 = full_range_start;
            range_end_1 = range_end;
        end
    end
end

% === Extract Segments for Templates ===
interpulse_len = (trigs_end(1) - trigs_beg(1) + 1) + 2 * template_leeway;
chn_data = nan(NSTIM, interpulse_len);

for i = 1:NSTIM
    seg_start = trigs_beg(i) - template_leeway;
    seg_end = trigs_end(i) + template_leeway;

    if seg_start < 1 || seg_end > length(amplifier_data)
        warning('Pulse %d out of bounds after padding. Skipping.', i);
        continue;
    end

    chn_data(i, :) = amplifier_data(seg_start:seg_end);
end

% === Generate Artifact Templates ===
template = generate_template(chn_data, template_mode, num_pulse, repeat_boundaries, buffer, period_avg, pca_k, window_size);

% === Subtract Templates ===
for i = 1:NSTIM
    seg_start = trigs_beg(i) - template_leeway;
    seg_end = trigs_end(i) + template_leeway;

    if seg_start < 1 || seg_end > length(amplifier_data)
        warning('Segment %d out of bounds. Skipping template subtraction.', i);
        continue;
    end

    segment = seg_start:seg_end;
    amplifier_data(segment) = amplifier_data(segment) - template(i, :);
end

% === Post-subtraction Inter-Pulse Drift Correction ===
for i = 1:NSTIM-1
    end_i = trigs_end(i);
    beg_next = trigs_beg(i+1);

    mid1 = round((end_i + trigs_beg(i)) / 2);
    mid2 = round((beg_next + trigs_end(i+1)) / 2);

    if mid1 >= mid2 || mid2 > length(amplifier_data)
        continue;
    end

    seg_idx = mid1:mid2;
    interpulse_len = length(seg_idx);

    template_end_val = amplifier_data(trigs_end(i));
    target_offset = -template_end_val;

    % === Compute ramp manually without linspace ===
    % Equivalent to: drift = linspace(0, target_offset, interpulse_len);
    idx = (0:interpulse_len-1)';                % column vector
    drift = (idx / (interpulse_len - 1)) * target_offset;

    amplifier_data(seg_idx) = amplifier_data(seg_idx) + drift';
end

amplifier_data_cleaned = amplifier_data;

% === Plotting ===
if exist('t_ms_1', 'var') && chan == plot_chan
    filtered_block = amplifier_data_cleaned(full_range_start_1:range_end_1);
    figure('Name', sprintf('Drift + Subtraction Summary (Chan %d)', chan), 'Position', [100 100 1100 450]); hold on;
    plot(t_ms_1, raw_block_1, 'Color', [0.8 0.3 0.3 0.4], 'DisplayName', 'Raw');
    plot(t_ms_1, med_filtered_1, 'Color', [0.2 0.5 0.8 0.5], 'DisplayName', 'Median Filtered');
    plot(t_ms_1, gauss_filtered_1, 'k-', 'LineWidth', 1.2, 'DisplayName', 'Med+Gauss Drift');
    plot(t_ms_1, filtered_block, 'Color', [0.2 0.6 0.9], 'DisplayName', 'Cleaned Output');
    for i = 1:NSTIM
        seg_start = trigs_beg(i) - template_leeway;
        seg_end = trigs_end(i) + template_leeway;
        if seg_start >= full_range_start_1 && seg_end <= range_end_1
            t_start = (seg_start - full_range_start_1) / fs * 1000 + t_ms_1(1);
            t_end = (seg_end - full_range_start_1) / fs * 1000 + t_ms_1(1);
            y_val = filtered_block(round((seg_start - full_range_start_1) + 1));
            y_val2 = filtered_block(round((seg_end - full_range_start_1) + 1));
            scatter(t_start, y_val, 20, 'b', 'filled', 'HandleVisibility', 'off');
            scatter(t_end, y_val2, 20, 'r', 'filled', 'HandleVisibility', 'off');
        end
    end
    xlabel('Time (ms)'); ylabel('Amplitude (\muV)');
    title(sprintf('Drift Removal + Template Subtraction - Chan %d', chan));
    legend('Location', 'best'); box off;
end
end