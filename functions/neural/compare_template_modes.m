function [cleaned_versions, trigs, stim_drift_all] = compare_template_modes(trial_path, template_params, plot_chan, template_modes, compare_plot)
% Compares artifact removal using different template subtraction modes.

% === Locate files ===
neural_file = fullfile(trial_path, 'neural_data.mat');
stim_file   = fullfile(trial_path, 'stim_data.mat');

if ~isfile(neural_file) || ~isfile(stim_file)
    warning('Missing neural or stim data in %s. Skipping.', trial_path);
    cleaned_versions = [];
    trigs = [];
    stim_drift_all = {};
    return;
end

% === Load full neural data into memory ===
neural_struct = load(neural_file);
if isfield(neural_struct, 'neural_data')
    neural_data = neural_struct.neural_data;
else
    warning('No neural_data field found in %s.', trial_path);
    cleaned_versions = [];
    trigs = [];
    stim_drift_all = {};
    return;
end
nChans   = size(neural_data, 1);
nSamples = size(neural_data, 2);

% === Load stim_data ===
stim_struct = load(stim_file);
if isfield(stim_struct, 'Stim_data')
    stim_data = stim_struct.Stim_data;
elseif isfield(stim_struct, 'stim_data')
    stim_data = stim_struct.stim_data;
else
    warning('No recognized stim_data field in %s. Skipping.', trial_path);
    cleaned_versions = [];
    trigs = [];
    stim_drift_all = {};
    return;
end

% === Detect stim channels ===
fs = 30000;
STIM_CHANS = find(any(stim_data ~= 0, 2));
if isempty(STIM_CHANS)
    disp('No stim signal detected.');
    cleaned_versions = [];
    trigs = [];
    stim_drift_all = {};
    return;
end

% === Trigger detection ===
TRIGDAT = stim_data(STIM_CHANS(1), :)';
trigs1 = find(diff(TRIGDAT) < 0);
trigs2 = find(diff(TRIGDAT) > 0);
trigs_rz = arrayfun(@(idx) ...
    idx + find(TRIGDAT(idx+1:end) == 0, 1, 'first'), ...
    trigs1, 'UniformOutput', false);
trigs_rz = cell2mat(trigs_rz(~cellfun('isempty', trigs_rz)));

trigs_beg = trigs1;
if length(trigs2) > length(trigs1)
    trigs_beg = trigs2;
end
trigs_beg = trigs_beg(1:2:end);
trigs_end = trigs_rz(2:2:end);

n_trigs = min(length(trigs_beg), length(trigs_end));
trigs = [trigs_beg(1:n_trigs), trigs_end(1:n_trigs)];
template_params.NSTIM = size(trigs, 1);

% === Preallocate ===
nModes = numel(template_modes);
cleaned_versions = zeros(nChans, nModes, nSamples);
stim_drift_all = cell(nChans, nModes);

% === Parallel Artifact Removal Loop ===
if isempty(gcp('nocreate'))
    parpool('local');
end

parfor chan = 1:nChans
    raw_trace = neural_data(chan, :);

    local_cleaned_versions = zeros(nModes, nSamples);
    local_stim_drift_all = cell(1, nModes);

    for m = 1:nModes
        mode = template_modes{m};
        [cleaned_trace, stim_drift] = ...
            template_subtraction(raw_trace, trigs, chan, template_params, mode, plot_chan);

        local_cleaned_versions(m, :) = cleaned_trace;
        local_stim_drift_all{m} = stim_drift;
    end

    cleaned_versions(chan, :, :) = local_cleaned_versions;
    stim_drift_all(chan, :) = local_stim_drift_all;
end


% === Optional Plotting ===
if compare_plot
    if nargin < 4 || isempty(plot_chan)
        plot_chan = 1;
    end

    raw_trace = neural_data(plot_chan, :);  % Full trace for plot

    % Optional: reapply notch filter
    notch_freq = 60;
    d = designfilt('bandstopiir', ...
        'FilterOrder', 4, ...
        'HalfPowerFrequency1', notch_freq - 2, ...
        'HalfPowerFrequency2', notch_freq + 2, ...
        'SampleRate', fs);
    raw_trace = filter(d, raw_trace);

    time_ms = (0:nSamples-1) / fs * 1000;
    figure('Name', sprintf('Template Subtraction Modes - Channel %d', plot_chan), ...
        'Position', [100 100 1200 500]);

    plot(time_ms, raw_trace, ...
        'Color', [0.8 0.3 0.3 0.4], 'LineWidth', 0.75, 'DisplayName', 'Original'); hold on;

    for m = 1:nModes
        mode_name = capitalize(template_modes{m});
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
