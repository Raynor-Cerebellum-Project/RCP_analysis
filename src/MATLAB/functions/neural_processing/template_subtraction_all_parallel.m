function [cleaned_versions, block2] = template_subtraction_all_parallel( ...
    neural_data, template_params, template_modes, trigs, repeat_boundaries, trial_path, plot_chan)

if nargin < 7
    plot_chan = -1;
end

[nChans, nSamples] = size(neural_data);

% === Determine number of repeats and pulses ===
num_repeats = numel(repeat_boundaries) - 1;
num_pulse   = size(trigs, 1) / num_repeats;

% === Preallocate ===
cleaned_versions = zeros(nChans, nSamples);
stim_drift_all   = cell(nChans);

% === Parallel Artifact Removal Loop ===
parfor chan = 1:nChans
    raw_trace = neural_data(chan, :);
    local_cleaned_trace = zeros(1, nSamples);

    mode = template_modes{1};

    if plot_chan == chan
        [cleaned_trace, stim_drift, block2] = template_subtraction_plot_before_after( ...
            raw_trace, trigs, template_params, mode, ...
            repeat_boundaries, num_repeats, num_pulse, chan);
    else
        [cleaned_trace, stim_drift] = template_subtraction( ...
            raw_trace, trigs, template_params, mode, ...
            repeat_boundaries, num_repeats, num_pulse);
            block2 = [];
    end


    local_cleaned_trace(:) = cleaned_trace;  % since there's only one mode
    local_stim_drift_all   = stim_drift;

    cleaned_versions(chan, :) = local_cleaned_trace;
    stim_drift_all{chan} = local_stim_drift_all;

end
% === Conditionally save if any drift data exists ===
if any(cellfun(@(c) ~isempty(c), stim_drift_all(:)))
    save(fullfile(trial_path, 'stim_drift_all.mat'), 'stim_drift_all', '-v7.3');
end
end
