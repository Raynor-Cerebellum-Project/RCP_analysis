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
%% === Identify Trial Groups ===
time_diffs = diff(trigs_beg);
repeat_gap_threshold = 2 * (2 * buffer + 1);
repeat_boundaries = [0; find(time_diffs > repeat_gap_threshold); numel(trigs_beg)];
num_repeats = numel(repeat_boundaries) - 1;
num_pulse = NSTIM / num_repeats;


%% === Non-Parametric Drift Removal: Median + Gaussian Filter ===
amplifier_data_drift_removed = amplifier_data;
stim_drift = struct('values', cell(num_repeats, 1), ...
    'indices', cell(num_repeats, 1));

for r = 1:num_repeats
    idx = (repeat_boundaries(r)+1):repeat_boundaries(r+1);
    full_range_start = trigs_end(idx(1))-4;  % after first pulse

    if r < num_repeats
        % Stop just before the *first* trig_beg of the next block
        range_end = trigs_beg(repeat_boundaries(r+1)+1) - 1;
    else
        % For the last block, go to the end of the last pulse + buffer
        range_end = trigs_end(idx(end)) + buffer;
    end

    if full_range_start > 0 && range_end <= length(amplifier_data)
        raw_block = double(amplifier_data(full_range_start:range_end));

        % === Median Filter (removes pulse spikes) ===
        med_filtered = medfilt1(raw_block, med_filt_range);

        % === Gaussian Filter (removes slow drift) ===
        gauss_filtered = imgaussfilt(med_filtered, gauss_filt_range);

        % === Subtract estimated drift ===
        amplifier_data_drift_removed(full_range_start:range_end) = ...
            amplifier_data_drift_removed(full_range_start:range_end) - gauss_filtered;
        stim_drift(r).values = gauss_filtered(:);  % store drift trace
        stim_drift(r).indices = full_range_start:range_end;  % corresponding sample indices

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

%% === Extract Segments for Templates ===
segment_len = (trigs_end(1) - trigs_beg(1) + 1) + 2 * template_leeway;
chn_data = nan(NSTIM, segment_len);

for i = 1:NSTIM
    seg_start = trigs_beg(i) - template_leeway;
    seg_end = trigs_end(i) + template_leeway;

    if seg_start < 1 || seg_end > length(amplifier_data_drift_removed)
        warning('Pulse %d out of bounds after padding. Skipping.', i);
        continue;
    end

    chn_data(i, :) = amplifier_data_drift_removed(seg_start:seg_end);
end

%% === Generate Artifact Templates ===
% === Initialize template ===
template = zeros(size(chn_data));

switch lower(template_mode)
    case 'local'
        for i = 1:NSTIM
            a = floor((i - 1) / num_pulse);
            idx = (1 + num_pulse * a):(min(num_pulse * (a + 1), NSTIM));
            local = movmean(chn_data(idx, :), window_size, 1);  % smooth full pulse
            base = local(i - num_pulse * a, :);
            template(i, :) = base - base(1);  % no taper, full subtraction
        end
    case 'local_drift_corr'
        first_pulse_indices = find(mod((1:NSTIM) - 1, num_pulse) == 0);  % all block-start indices

        for i = 1:NSTIM
            a = floor((i - 1) / num_pulse);  % block index
            idx = (1 + num_pulse * a):(min(num_pulse * (a + 1), NSTIM));  % pulses in current block
            relative_idx = i - num_pulse * a;  % pulse index within block

            % === Handle first pulse of a block ===
            if relative_idx == 1
                % Find all previous first-pulse indices before this one
                prev_first_pulses = first_pulse_indices(first_pulse_indices < i);
                n_available = numel(prev_first_pulses);

                if n_available >= 1
                    % Use up to the last 3 previous first-pulse templates
                    n_use = min(3, n_available);
                    ref_idxs = prev_first_pulses(end - n_use + 1:end);
                    base = mean(chn_data(ref_idxs, :), 1);
                    template_i = base - base(1);  % baseline align
                    template_end_val = template_i(end);

                    target_offset = -template_end_val;
                    drift = linspace(0, target_offset, segment_len);
                    template_i = template_i + drift;

                    template(i, :) = template_i;
                else
                    continue;  % Very first pulse — skip
                end

                continue  % Done for first pulse
            end

            % === Build local history window excluding first pulses ===
            prev_pulses = [];
            count = 0;
            j = i - 1;

            while j > 0 && count < window_size
                prev_rel_idx = mod(j - 1, num_pulse) + 1;
                if prev_rel_idx ~= 1
                    prev_pulses = [chn_data(j, :); prev_pulses];
                    count = count + 1;
                end
                j = j - 1;
            end

            if ~isempty(prev_pulses)
                base = mean(prev_pulses, 1);
            else
                base = zeros(1, segment_len);
            end

            template_i = base - base(1);  % baseline align
            template_end_val = template_i(end);

            target_offset = -template_end_val;
            drift = linspace(0, target_offset, segment_len);
            template_i = template_i + drift;

            template(i, :) = template_i;
        end
    case 'carryover'
        for b = 1:num_repeats
            idx = (repeat_boundaries(b)+1):repeat_boundaries(b+1);
            if b > 1
                prev_idx = (repeat_boundaries(b-1)+1):repeat_boundaries(b);
                base = mean(chn_data(prev_idx, 1:(period_avg + buffer)), 1);
            else
                base = mean(chn_data(idx, 1:(period_avg + buffer)), 1);
            end
            base = base - base(1);
            taper = linspace(base(end), 0, segment_len - (period_avg + buffer) + 1);
            for i = idx
                template(i, 1:(period_avg + buffer)) = base;
                template(i, period_avg + buffer:end) = taper;
            end
        end
    case 'pca'
        % === Identify first vs non-first pulses ===
        first_mask = mod((1:NSTIM) - 1, num_pulse) == 0;
        X_first = chn_data(first_mask, :);
        X_nonfirst = chn_data(~first_mask, :);

        k = pca_k;  % typically 3
        template = zeros(size(chn_data));

        % === PCA for first-pulse segments ===
        if size(X_first, 1) >= k
            [coeff_f, score_f, ~, ~, ~, mu_f] = pca(X_first);
            coeff_f = coeff_f(:, 1:k);
            score_f = score_f(:, 1:k);
            recon_f = score_f * coeff_f' + mu_f;
        else
            warning('Not enough first-pulse segments for PCA (%d found)', size(X_first,1));
            recon_f = zeros(sum(first_mask), segment_len);
        end

        % === PCA for non-first segments ===
        if size(X_nonfirst, 1) >= k
            [coeff_n, score_n, ~, ~, ~, mu_n] = pca(X_nonfirst);
            coeff_n = coeff_n(:, 1:k);
            score_n = score_n(:, 1:k);
            recon_n = score_n * coeff_n' + mu_n;
        else
            warning('Not enough non-first-pulse segments for PCA (%d found)', size(X_nonfirst,1));
            recon_n = zeros(sum(~first_mask), segment_len);
        end

        % === Assign reconstructed templates with drift correction ===
        first_counter = 1;
        nonfirst_counter = 1;

        for i = 1:NSTIM
            if first_mask(i)
                temp_i = recon_f(first_counter, :); first_counter = first_counter + 1;
            else
                temp_i = recon_n(nonfirst_counter, :); nonfirst_counter = nonfirst_counter + 1;
            end

            temp_i = temp_i - temp_i(1);  % baseline align
            adj_avg = mean(temp_i(end-4:end));
            drift = linspace(0, -adj_avg, segment_len);
            temp_i = temp_i + drift;

            template(i, :) = temp_i;
        end

        % === Optional plots ===
        if chan == 0
            figure; hold on;
            t_ms = (0:segment_len-1) / 30;
            colors = lines(k);

            if exist('coeff_f', 'var')
                subplot(1,2,1); hold on; title('First Pulse PCs');
                for j = 1:k
                    plot(t_ms, coeff_f(:,j), 'Color', colors(j,:), 'LineWidth', 2);
                end
                xlabel('Time (ms)'); ylabel('PC Weight');
            end

            if exist('coeff_n', 'var')
                subplot(1,2,2); hold on; title('Non-First Pulse PCs');
                for j = 1:k
                    plot(t_ms, coeff_n(:,j), 'Color', colors(j,:), 'LineWidth', 2);
                end
                xlabel('Time (ms)'); ylabel('PC Weight');
            end
        end

    otherwise
        error('Invalid template_mode. Use ''local'', ''global'', ''carryover'', or ''pca''.');
end
%% === Subtract Templates ===
amplifier_data_cleaned = amplifier_data_drift_removed;
for i = 1:NSTIM
    seg_start = trigs_beg(i) - template_leeway;
    seg_end = trigs_end(i) + template_leeway;
    
    if seg_start < 1 || seg_end > length(amplifier_data_cleaned)
        warning('Segment %d out of bounds. Skipping template subtraction.', i);
        continue;
    end
    
    segment = seg_start:seg_end;
    amplifier_data_cleaned(segment) = amplifier_data_cleaned(segment) - template(i, :);
end
%% === Post-subtraction Inter-Pulse Drift Correction ===
amplifier_data_cleaned_post = amplifier_data_cleaned;

for i = 1:NSTIM-1
    % Define midpoint between pulses
    end_i = trigs_end(i);
    beg_next = trigs_beg(i+1);

    mid1 = round((end_i + trigs_beg(i)) / 2);
    mid2 = round((beg_next + trigs_end(i+1)) / 2);

    if mid1 >= mid2 || mid2 > length(amplifier_data_cleaned)
        continue;
    end

    % Define segment and segment length
    seg_idx = mid1:mid2;
    segment_len = length(seg_idx);

    % Get end value of previous pulse's template-subtracted trace
    if trigs_end(i) <= length(amplifier_data_cleaned)
        template_end_val = amplifier_data_cleaned(trigs_end(i));
    else
        template_end_val = 0;
    end

    % Create linear drift vector
    target_offset = -template_end_val;
    drift = linspace(0, target_offset, segment_len);

    % Subtract drift from inter-pulse segment
    amplifier_data_cleaned_post(seg_idx) = amplifier_data_cleaned_post(seg_idx) + drift;
end

amplifier_data_cleaned = amplifier_data_cleaned_post;


% === Plot decay fit for first repeat ===
if exist('t_ms_1', 'var') && chan == plot_chan
    filtered_block = amplifier_data_cleaned(full_range_start_1:range_end_1);

    figure('Name', sprintf('Drift + Subtraction Summary (Chan %d)', chan), ...
        'Position', [100 100 1100 450]); hold on;

    plot(t_ms_1, raw_block_1, 'Color', [0.8 0.3 0.3 0.4], 'DisplayName', 'Raw');
    plot(t_ms_1, med_filtered_1, 'Color', [0.2 0.5 0.8 0.5], 'DisplayName', 'Median Filtered');
    plot(t_ms_1, gauss_filtered_1, 'k-', 'LineWidth', 1.2, 'DisplayName', 'Med+Gauss Drift');
    plot(t_ms_1, filtered_block, 'Color', [0.2 0.6 0.9], 'DisplayName', 'Cleaned Output');
    % === Mark padded segment borders with scatter points ===
    hold on;
    for i = 1:NSTIM
        seg_start = trigs_beg(i) - template_leeway;
        seg_end = trigs_end(i) + template_leeway;
        if seg_start >= full_range_start_1 && seg_end <= range_end_1
            t_start = (seg_start - full_range_start_1) / fs * 1000 + t_ms_1(1);
            t_end = (seg_end - full_range_start_1) / fs * 1000 + t_ms_1(1);
            y_val = filtered_block(round((seg_start - full_range_start_1) + 1));  % corresponding amplitude
            y_val2 = filtered_block(round((seg_end - full_range_start_1) + 1));    % amplitude at end

            scatter(t_start, y_val, 20, 'b', 'filled', 'HandleVisibility', 'off');
            scatter(t_end, y_val2, 20, 'r', 'filled', 'HandleVisibility', 'off');
        end
    end
    xlabel('Time (ms)'); ylabel('Amplitude (µV)');
    title(sprintf('Drift Removal + Template Subtraction - Chan %d', chan));
    legend('Location', 'best'); box off;
end



%% === Apply Notch Filter to Remove 60 Hz ===
% notch_freq = 60;
% d = designfilt('bandstopiir', 'FilterOrder', 4, 'HalfPowerFrequency1', notch_freq - 10, 'HalfPowerFrequency2', notch_freq + 10, 'SampleRate', fs);
% for r = 1:num_repeats
%     idx = (repeat_boundaries(r)+1):repeat_boundaries(r+1);
%     trig_range = trigs(idx);
%     range_start = trig_range(1) - buffer;
%     range_end = trig_range(end) + buffer;
%     if range_start > 0 && range_end <= length(amplifier_data_copy)
%         amplifier_data_copy(range_start:range_end) = filter(d, amplifier_data_copy(range_start:range_end));
%     end
% end
end
function y_scaled = rescale_fit_to_bounds(y, y_start, y_end)
y0 = y(1);
y1 = y(end);
n = numel(y);
w = linspace(0, 1, n);
y_shift = (1 - w) * (y0 - y_start) + w * (y1 - y_end);
y_scaled = y - y_shift;
end