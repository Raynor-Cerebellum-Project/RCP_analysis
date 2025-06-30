function cleaned_versions = compare_local_windows(amplifier_data, stim_data, template_params, plot_chan)

    fs = 30000;
    STIM_CHANS = find(any(stim_data ~= 0, 2));

    if isempty(STIM_CHANS)
        disp('No stim signal detected.');
        cleaned_versions = [];
        return;
    end

    % Detect stim triggers
    TRIGDAT = stim_data(STIM_CHANS(1), :)';
    trigs1 = find(diff(TRIGDAT) < 0);
    trigs2 = find(diff(TRIGDAT) > 0);
    trigs = trigs1;
    if length(trigs2) > length(trigs1)
        trigs = trigs2;
    end
    trigs = trigs(1:2:end);
    NSTIM = length(trigs);
    template_params.NSTIM = NSTIM;

    % Parameters
    window_sizes = [1, 3, 5, 7];
    nChans = size(amplifier_data, 1);
    nSamples = size(amplifier_data, 2);
    nWindows = length(window_sizes);

    % Preallocate
    cleaned_versions = zeros(nChans, nWindows, nSamples);

    % Loop through all channels and window sizes
    for chan = 1:nChans
        raw_trace = amplifier_data(chan, :);

        for w = 1:nWindows
            template_params.movmean_window = window_sizes(w);
            cleaned_versions(chan, w, :) = template_subtraction(raw_trace, trigs, chan, template_params, 'local');
        end
    end

    % Plot results for selected channel in ms
    if nargin < 4 || isempty(plot_chan)
        plot_chan = 1;  % Default to channel 1
    end

    raw_trace = amplifier_data(plot_chan, :);
    time_ms = (0:nSamples-1) / fs * 1000;  % Convert samples to milliseconds

    figure('Name', sprintf('Template Subtraction Comparison - Channel %d', plot_chan), ...
           'Position', [100 100 1200 500]);
    plot(time_ms, raw_trace, 'Color', [0.2 0.8 0.2 0.4], 'DisplayName', 'Original'); hold on;

    colors = lines(nWindows);
    for w = 1:nWindows
        plot(time_ms, squeeze(cleaned_versions(plot_chan, w, :)), 'Color', colors(w, :), 'LineWidth', 0.8, ...
             'DisplayName', sprintf('Local Avg (%d pulses)', window_sizes(w)));
    end

    legend;
    xlabel('Time (ms)');
    ylabel('Amplitude');
    title(sprintf('Artifact Removal with Varying Local Window Sizes (Channel %d)', plot_chan));r
end
