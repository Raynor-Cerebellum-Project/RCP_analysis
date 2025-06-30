function [cleaned_versions, trigs, stim_drift_all] = compare_template_modes(amplifier_data, stim_data, template_params, plot_chan, template_modes, compare_plot)
% Compares artifact removal using different template subtraction modes.
%
% Modes: 'global', 'carryover', etc.

fs = 30000;  % Sampling rate (Hz)
STIM_CHANS = find(any(stim_data ~= 0, 2));

if isempty(STIM_CHANS)
    disp('No stim signal detected.');
    cleaned_versions = [];
    trigs = [];
    stim_drift_all = {};
    return;
end

% === Apply causal 60 Hz notch filter to amplifier_data ===
notch_freq = 60;  % Hz
d = designfilt('bandstopiir', ...
    'FilterOrder', 4, ...
    'HalfPowerFrequency1', notch_freq - 2, ...
    'HalfPowerFrequency2', notch_freq + 2, ...
    'SampleRate', fs);

for ch = 1:size(amplifier_data, 1)
    amplifier_data(ch, :) = filter(d, amplifier_data(ch, :));  % causal filter
end

% === Detect triggers ===
TRIGDAT = stim_data(STIM_CHANS(1), :)';
trigs1 = find(diff(TRIGDAT) < 0);  % falling edges
trigs2 = find(diff(TRIGDAT) > 0);  % rising edges

% Detect return-to-zero after each falling edge
trigs_rz = arrayfun(@(idx) ...
    idx + find(TRIGDAT(idx+1:end) == 0, 1, 'first'), ...
    trigs1, 'UniformOutput', false);
trigs_rz = cell2mat(trigs_rz(~cellfun('isempty', trigs_rz)));

% Select trigger edges
trigs_beg = trigs1;
if length(trigs2) > length(trigs1)
    trigs_beg = trigs2;
end
trigs_beg = trigs_beg(1:2:end);
trigs_end = trigs_rz(2:2:end);

% Final triggers
n_trigs = min(length(trigs_beg), length(trigs_end));
trigs = [trigs_beg(1:n_trigs), trigs_end(1:n_trigs)];
template_params.NSTIM = size(trigs, 1);

% === Init ===
nModes = numel(template_modes);
nChans = size(amplifier_data, 1);
nSamples = size(amplifier_data, 2);
cleaned_versions = zeros(nChans, nModes, nSamples);
stim_drift_all = cell(nChans, nModes);

% === Artifact removal loop ===
for chan = 1:nChans
    raw_trace = amplifier_data(chan, :);
    for m = 1:nModes
        mode = template_modes{m};
        [cleaned_trace, stim_drift] = ...
            template_subtraction(raw_trace, trigs, chan, template_params, mode, plot_chan);

        cleaned_versions(chan, m, :) = cleaned_trace;
        stim_drift_all{chan, m} = stim_drift;
    end
end

% === Optional comparison plot ===
if compare_plot
    if nargin < 4 || isempty(plot_chan)
        plot_chan = 1;
    end

    raw_trace = amplifier_data(plot_chan, :);
    time_ms = (0:nSamples-1) / fs * 1000;

    figure('Name', sprintf('Template Subtraction Modes - Channel %d', plot_chan), ...
        'Position', [100 100 1200 500]);

    % === Original trace ===
    plot(time_ms, raw_trace, ...
        'Color', [0.8 0.3 0.3 0.4], 'LineWidth', 0.75, 'DisplayName', 'Original'); hold on;

    for m = 1:nModes
        mode_name = capitalize(template_modes{m});

        % === Composite drift trace (NaN elsewhere) ===
        drift_trace = nan(size(raw_trace));
        if ~isempty(stim_drift_all{plot_chan, m})
            drift = stim_drift_all{plot_chan, m};
            for r = 1:numel(drift)
                indices = drift(r).indices;
                values = drift(r).values;
                if all(indices <= length(drift_trace))
                    drift_trace(indices) = values(:)';
                end
            end

            plot(time_ms, drift_trace, ...
                'Color', [0.3 0.3 0.3], 'LineWidth', 0.6, ...
                'DisplayName', sprintf('%s Drift', mode_name));
        end

        % === Plot final cleaned trace ===
        plot(time_ms, squeeze(cleaned_versions(plot_chan, m, :)), ...
            'Color', [0.2 0.6 0.9], 'LineWidth', 0.8, ...
            'DisplayName', sprintf('%s Cleaned', mode_name));
    end

    legend('Location', 'best');
    xlabel('Time (ms)');
    ylabel('Amplitude (µV)');
    title(sprintf('Artifact Removal using Different Template Modes (Channel %d)', plot_chan));
    box off;
end



end

function str = capitalize(s)
s = strrep(s, '_', ' ');
str = [upper(s(1)), s(2:end)];
end
