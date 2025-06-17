function amplifier_data_copy = template_subtraction(amplifier_data, trigs, chan, params, template_mode, tracePlot)
% Subtracts stimulus-locked artifacts from electrophysiology data
% template_mode: 'local', 'global', 'pca', or 'carryover'

NSTIM = params.NSTIM;
start = params.start;
isstim = params.isstim;
period = trigs(2) - trigs(1);
period_avg = params.period_avg;
prebuffer = params.buffer;
skip_n = params.skip_n;
window_size = params.movmean_window;
pca_k = params.pca_components;

% Extract aligned segments
chn_data = zeros(NSTIM, period + prebuffer);
for i = 1:NSTIM
    segment = (-prebuffer+1:period) + trigs(i);
    chn_data(i, :) = amplifier_data(segment);
end

% Identify trial groups
time_diffs = diff(trigs);
repeat_gap_threshold = period * 2;
repeat_boundaries = [0; find(time_diffs > repeat_gap_threshold); numel(trigs)];
num_repeats = numel(repeat_boundaries) - 1;
num_pulse = NSTIM / num_repeats;

% Determine global template indices
temp = start:NSTIM;
temp = temp(mod(temp, num_pulse) == 0 | mod(temp, num_pulse) > skip_n);

% === Compute Template ===
template = zeros(size(chn_data));

switch lower(template_mode)
    case 'local'
        for i = 1:NSTIM
            a = floor((i - 1) / num_pulse);
            idx = (1 + num_pulse * a):(min(num_pulse * (a + 1), NSTIM));
            local = movmean(chn_data(idx, 1:period_avg+prebuffer), window_size);
            base = local(i - num_pulse * a, :);
            template(i, 1:period_avg+prebuffer) = base - base(1);
            template(i, period_avg+prebuffer:end) = ...
                linspace(base(end), 0, period - period_avg + 1);
        end

    case 'global'
        global_mean = mean(chn_data(temp, 1:period_avg+prebuffer), 1);
        baseline_adj = global_mean - global_mean(1);
        taper = linspace(global_mean(end), 0, period - period_avg + 1);
        for i = 1:NSTIM
            template(i, 1:period_avg+prebuffer) = baseline_adj;
            template(i, period_avg+prebuffer:end) = taper;
        end

    case 'carryover'
        for b = 1:num_repeats
            idx = (repeat_boundaries(b)+1):repeat_boundaries(b+1);
            if b > 1
                prev_idx = (repeat_boundaries(b-1)+1):repeat_boundaries(b);
                base = mean(chn_data(prev_idx, 1:period_avg+prebuffer), 1);
            else
                base = mean(chn_data(idx, 1:period_avg+prebuffer), 1);
            end
            base = base - base(1);
            taper = linspace(base(end), 0, period - period_avg + 1);
            for i = idx
                template(i, 1:period_avg+prebuffer) = base;
                template(i, period_avg+prebuffer:end) = taper;
            end
        end

    case 'pca'
        [coeff, score, mu] = pca(chn_data);
        template = score(:, 1:pca_k) * coeff(:, 1:pca_k)' + mu;

    otherwise
        error('Invalid template_mode. Use ''local'', ''global'', ''carryover'', or ''pca''.');
end

% === Subtract Template ===
amplifier_data_copy = amplifier_data;
for i = 1:NSTIM
    segment = (-prebuffer+1:period) + trigs(i);
    amplifier_data_copy(segment) = amplifier_data_copy(segment) - template(i, :);
end

% === Optional Plot ===
if tracePlot && (chan == 1 || chan == 16)
    t_ms = ((-prebuffer+1):period) / 30;
    chn_zeroed = chn_data - chn_data(:,1);
    template_zeroed = template - template(:,1);
    cleaned_zeroed = zeros(size(template));
    for i = 1:NSTIM
        segment = (-prebuffer+1:period) + trigs(i);
        cleaned_zeroed(i, :) = amplifier_data_copy(segment) - amplifier_data_copy(segment(1));
    end

    cmap = jet(NSTIM);
    figure('Name', sprintf('Before vs After Subtraction (Chan %d)', chan), ...
        'Position', [100 100 800 600]);

    subplot(3,1,1); hold on;
    for i = 1:NSTIM
        plot(t_ms, chn_zeroed(i,:), 'Color', [cmap(i,:), 0.5]);
    end
    plot(t_ms, mean(template_zeroed, 1), 'k--', 'LineWidth', 2);
    title(sprintf('Original & Template (Chan %d - %s)', chan, upper(template_mode)));
    xlabel('Time (ms)'); ylabel('Amplitude'); box off;

    subplot(3,1,2); hold on;
    for i = 1:NSTIM
        plot(t_ms, template_zeroed(i,:), 'Color', [cmap(i,:), 0.7]);
    end
    title('Templates'); xlabel('Time (ms)'); box off;

    subplot(3,1,3); hold on;
    for i = 1:NSTIM
        plot(t_ms, cleaned_zeroed(i,:), 'Color', [cmap(i,:), 0.5]);
    end
    title('Cleaned'); xlabel('Time (ms)'); ylabel('Amplitude'); box off;
end
end

function disp_segment_info(~, ~, idx)
fprintf('You clicked on segment %d\n', idx);
end
