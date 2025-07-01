function cleaned_versions = sweep_template_param(amplifier_data, stim_data, chan, sweep_name, sweep_values, fixed_params, template_modes)
% Sweeps a single template parameter and overlays results from multiple template modes.
% One figure is generated per template mode with original trace and all sweep results.

    fs = 30000;
    nSweep = length(sweep_values);
    raw_trace = amplifier_data(chan, :);
    time_ms = (0:length(raw_trace)-1) / fs * 1000;
    nModes = numel(template_modes);
    compare_plot = true;

    % Store cleaned traces [sweep x mode x time]
    cleaned_versions = zeros(nSweep, nModes, length(raw_trace));

    % Sweep parameter
    for i = 1:nSweep
        val = sweep_values(i);
        template_params = fixed_params;
        template_params.(sweep_name) = val;

        cleaned = compare_template_modes(amplifier_data, stim_data, template_params, chan, template_modes, compare_plot);
        cleaned_versions(i, :, :) = squeeze(cleaned(chan, :, :));
    end

    % Colors
    % cmap = lines(nSweep);
    % 
    % % ---- PLOT: One figure per template mode ----
    % for m = 1:nModes
    %     mode = template_modes{m};
    %     figure('Name', sprintf('%s Sweep - %s - Channel %d', capitalize(sweep_name), capitalize(mode), chan), ...
    %            'Position', [100 100 1400 400]);
    % 
    %     % Plot original
    %     plot(time_ms, raw_trace, 'Color', [0.2 0.2 0.2 0.3], 'LineWidth', 1.2, 'DisplayName', 'Original'); hold on;
    % 
    %     % Plot sweep values
    %     for i = 1:nSweep
    %         plot(time_ms, squeeze(cleaned_versions(i, m, :)), 'LineWidth', 1.2, ...
    %              'Color', cmap(i, :), ...
    %              'DisplayName', sprintf('%s = %d', sweep_name, sweep_values(i)));
    %     end
    % 
    %     legend('Location', 'best');
    %     xlabel('Time (ms)');
    %     ylabel('Amplitude');
    %     title(sprintf('%s Template Subtraction - Sweep %s (Channel %d)', ...
    %                   capitalize(mode), capitalize(sweep_name), chan));
    % end
end

function str = capitalize(s)
    str = [upper(s(1)), s(2:end)];
end
