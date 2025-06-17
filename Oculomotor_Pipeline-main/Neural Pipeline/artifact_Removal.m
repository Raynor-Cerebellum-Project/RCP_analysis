function cleaned_local = artifact_removal(amplifier_data, stim_data, template_params)
% Performs artifact removal on multichannel electrophysiological data
% by identifying stimulation triggers and applying template subtraction.

    fs = 30000;  % Sampling rate in Hz (unused in this code but may be relevant for time conversion)

    STIM_CHANS = find(any(stim_data ~= 0, 2));

    if isempty(STIM_CHANS)
        disp('No stimulation detected.');
        return;
    end

    % Get stimulation triggers
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
    template_params.movmean_window = 3;

    % Loop through all channels
    for chan = 1:128
        raw_trace = amplifier_data(chan, :);

        % Apply local template subtraction
        cleaned_local = template_subtraction(raw_trace, trigs, chan, template_params, 'local');

        % Apply global template subtraction
        % cleaned_global = template_subtraction(raw_trace, trigs, chan, template_params, 'global');

        % % Plot original vs both cleaned versions
        % figure('Name', sprintf('Channel %d Artifact Subtraction Comparison', chan), 'Position', [100 100 1200 400]);
        % plot(raw_trace, 'Color', [0.2 0.8 0.2 0.5], 'DisplayName', 'Original'); hold on;
        % plot(cleaned_local, 'Color', [0.2 0.2 1 0.8], 'DisplayName', 'Local Template'); 
        % plot(cleaned_global, 'Color', [1 0.2 0.2 0.8], 'DisplayName', 'Global Template'); 
        % legend;
        % title(sprintf('Channel %d: Artifact Removal Comparison', chan));
        % xlabel('Samples'); ylabel('Amplitude');
        % drawnow;
    end
end
