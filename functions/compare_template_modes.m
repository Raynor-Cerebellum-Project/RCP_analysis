function [cleaned_versions, trigs] = compare_template_modes(amplifier_data, stim_data, template_params, plot_chan, template_modes, compare_plot)
% Compares artifact removal using different template subtraction modes.
%
% Modes: 'global', 'carryover'

fs = 30000;  % Sampling rate (Hz)
STIM_CHANS = find(any(stim_data ~= 0, 2));

if isempty(STIM_CHANS)
    disp('No stim signal detected.');
    cleaned_versions = [];
    return;
end
tracePlot = false;
% Detect triggers
TRIGDAT = stim_data(STIM_CHANS(1), :)';
trigs1 = find(diff(TRIGDAT) < 0);
trigs2 = find(diff(TRIGDAT) > 0);
clear TRIGDAT;
trigs = trigs1;
if length(trigs2) > length(trigs1)
    trigs = trigs2;
end
trigs = trigs(1:2:end);  % Downsample to remove second falling edge (need to adjust if we change waveforms)
NSTIM = length(trigs);
template_params.NSTIM = NSTIM;

% Template modes to compare
nModes = numel(template_modes);
nChans = size(amplifier_data, 1);
nSamples = size(amplifier_data, 2);

output_dir = fullfile(tempdir, 'template_cleaned_temp');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

if isempty(gcp('nocreate'))
    parpool(6);  % or specify cores with parpool(4), etc.
end

parfor chan = 1:nChans
    raw_trace = amplifier_data(chan, :);
    cleaned_chan = zeros(nModes, nSamples, 'like', raw_trace);

    for m = 1:nModes
        mode = template_modes{m};
        cleaned_chan(m, :) = template_subtraction(raw_trace, trigs, chan, template_params, mode, tracePlot);
    end

    % Save using struct, correctly passing the variable (not a string)
    save(fullfile(output_dir, sprintf('cleaned_chan_%03d.mat', chan)), ...
         '-fromstruct', struct('cleaned_chan', cleaned_chan), '-v7.3');
end

if ~compare_plot
    clear amplifier_data;
end
cleaned_versions = zeros(nChans, nModes, nSamples);
for chan = 1:nChans
    fname = fullfile(output_dir, sprintf('cleaned_chan_%03d.mat', chan));
    S = load(fname);
    cleaned_versions(chan, :, :) = S.cleaned_chan;
    delete(fname);  % <---- Clean up temp file
end

clear S raw_trace cleaned_chan;
java.lang.System.gc;
% delete(gcp('nocreate'));
if compare_plot
    % Plot results for selected channel
    if nargin < 4 || isempty(plot_chan)
        plot_chan = 1;
    end

    raw_trace = amplifier_data(plot_chan, :);
    time_ms = (0:nSamples-1) / fs * 1000;

    figure('Name', sprintf('Template Subtraction Modes - Channel %d', plot_chan), ...
        'Position', [100 100 1200 500]);
    plot(time_ms, raw_trace, 'Color', [0.2 0.8 0.2 0.4], 'DisplayName', 'Original'); hold on;

    colors = lines(nModes);
    for m = 1:nModes
        plot(time_ms, squeeze(cleaned_versions(plot_chan, m, :)), ...
            'Color', colors(m, :), 'LineWidth', 1, ...
            'DisplayName', sprintf('%s Template', capitalize(template_modes{m})));
    end

    legend;
    xlabel('Time (ms)');
    ylabel('Amplitude');
    title(sprintf('Artifact Removal using Different Template Modes (Channel %d)', plot_chan));
end
end
function str = capitalize(s)
str = [upper(s(1)), s(2:end)];
end
