function [amplifier_data_cleaned, stim_drift] = template_subtraction( ...
    amplifier_data, trigs, params, template_mode, repeat_boundaries, num_repeats, num_pulse)

% === Unpack Parameters ===
NSTIM = params.NSTIM;
buffer = params.buffer;
template_leeway = params.template_leeway;
stim_neural_delay = params.stim_neural_delay;
window_size = params.movmean_window;
med_filt_range = params.med_filt_range;
gauss_filt_range = params.gauss_filt_range;
pca_k = params.pca_k;

trigs_beg = trigs(:, 1);
trigs_end = trigs(:, 2) + stim_neural_delay;

stim_drift = struct('values', cell(num_repeats, 1), 'range', cell(num_repeats, 1));

for r = 1:num_repeats
    idx = (repeat_boundaries(r)+1):repeat_boundaries(r+1);
    full_range_start = trigs_end(idx(1)) - 4;

    if r < num_repeats
        range_end = trigs_beg(repeat_boundaries(r+1)+1) - 1;
    else
        range_end = trigs_end(idx(end)) + buffer;
    end

    if full_range_start > 0 && range_end <= length(amplifier_data)
        raw_block = double(amplifier_data(full_range_start:range_end));
        med_filtered = movmedian(raw_block, med_filt_range, 'Endpoints', 'shrink');
        w = gausswin(gauss_filt_range, 2.5); w = w / sum(w);
        gauss_filtered = filtfilt(w, 1, med_filtered);

        amplifier_data(full_range_start:range_end) = amplifier_data(full_range_start:range_end) - gauss_filtered;

        stim_drift(r).values = single(gauss_filtered(:));
        stim_drift(r).range = [full_range_start, range_end];
    end
end

% === Template subtraction ===
interpulse_len = (trigs_end(1) - trigs_beg(1) + 1) + 2 * template_leeway;
chn_data = nan(NSTIM, interpulse_len);
valid_indices = false(NSTIM, 1);

% Single loop to extract valid segments and flag usable indices
for i = 1:NSTIM
    seg_start = trigs_beg(i) - template_leeway;
    seg_end   = trigs_end(i) + template_leeway;
    if seg_start >= 1 && seg_end <= length(amplifier_data)
        chn_data(i, :) = amplifier_data(seg_start:seg_end);
        valid_indices(i) = true;
    end
end

% Compute template
template = generate_template(chn_data, template_mode, repeat_boundaries, window_size, pca_k);

% Apply template subtraction using only valid indices
for i = find(valid_indices)'
    seg_start = trigs_beg(i) - template_leeway;
    seg_end   = trigs_end(i) + template_leeway;
    amplifier_data(seg_start:seg_end) = amplifier_data(seg_start:seg_end) - template(i, :);
end


% === Interpulse drift correction ===
for r = 1:num_repeats
    idx = (repeat_boundaries(r)+1):repeat_boundaries(r+1);
    for j = 1:(length(idx)-1)
        i = idx(j);      % current pulse
        i_next = idx(j+1);  % next pulse in the same block

        end_i = trigs_end(i);
        beg_next = trigs_beg(i_next);
        mid1 = round((end_i + trigs_beg(i)) / 2);
        mid2 = round((beg_next + trigs_end(i_next)) / 2);

        if mid1 >= mid2 || mid2 > length(amplifier_data), continue; end

        seg_idx = mid1:mid2;
        template_end_val = amplifier_data(trigs_end(i));
        interpulse_len = length(seg_idx);
        drift = (0:interpulse_len-1)' / (interpulse_len - 1) * -template_end_val;
        amplifier_data(seg_idx) = amplifier_data(seg_idx) + drift';
    end
end


amplifier_data_cleaned = amplifier_data;
end
