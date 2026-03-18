function stim_ext = extract_stim_triggers_and_blocks(stim_data)
% Extract stimulation pulse triggers and block boundaries from stim_data
%
% Input
% -----
% stim_data : numeric array, size (n_channels, n_samples)
%     Raw stim stream. Zero means baseline.
%
% Output
% ------
% stim_ext : struct with fields
%     .active_channels       (n_active_channels x 1)  1-based channel IDs
%     .trigger_pairs         (n_pulses x 2)           [start_sample, end_sample]
%     .block_bounds_samples  (n_blocks x 2)           [block_start_sample, block_end_sample]
%     .pulse_sizes           (n_pulses x 1)

    if ndims(stim_data) ~= 2
        error('stim_data must be a 2D array of size (n_channels, n_samples).');
    end

    % 1) Active channels
    active_channels = find(any(stim_data ~= 0, 2));

    if isempty(active_channels)
        stim_ext = struct( ...
            'active_channels', [], ...
            'trigger_pairs', zeros(0, 2), ...
            'block_bounds_samples', zeros(0, 2), ...
            'pulse_sizes', [] ...
        );
        return;
    end

    % Use first active channel for detection
    det_ch = active_channels(1);

    % In Python you added +1 to channel indices, but MATLAB is already 1-based
    stim_signal = double(stim_data(det_ch, :));

    if numel(stim_signal) < 2
        stim_ext = struct( ...
            'active_channels', active_channels(:), ...
            'trigger_pairs', zeros(0, 2), ...
            'block_bounds_samples', zeros(0, 2), ...
            'pulse_sizes', [] ...
        );
        return;
    end

    % 2) Edge detection
    d = diff(stim_signal);

    falling_edge = find(d < 0) + 1;
    rising_edge  = find(d > 0) + 1;

    % For each falling edge, find first subsequent return to zero
    rz = [];
    for k = 1:numel(falling_edge)
        idx = falling_edge(k);
        rel = find(stim_signal(idx:end) == 0, 1, 'first');
        if ~isempty(rel)
            rz(end+1, 1) = idx + rel - 1;
        end
    end

    beg = falling_edge;
    if numel(rising_edge) > numel(falling_edge)
        beg = rising_edge;
    end

    % Biphasic pulse handling
    beg  = beg(1:2:end);
    end_ = rz(2:2:end);

    n = min(numel(beg), numel(end_));
    if n == 0
        trigger_pairs = zeros(0, 2);
        pulse_sizes = [];
    else
        beg_use = beg(1:n);
        end_use = end_(1:n);
    
        beg_use = beg_use(:);
        end_use = end_use(:);
    
        trigger_pairs = [beg_use, end_use];
        pulse_sizes = trigger_pairs(:, 2) - trigger_pairs(:, 1);
    end

    % 3) Block boundaries
    if isempty(trigger_pairs)
        block_bounds_samples = zeros(0, 2);
    else
        pulse_size_ref = round(median(pulse_sizes));
        repeat_gap_threshold = 50 * pulse_size_ref;

        starts = trigger_pairs(:, 1);
        ends_  = trigger_pairs(:, 2);

        gaps = diff(starts);
        cut_points = find(gaps > repeat_gap_threshold) + 1;

        block_boundaries_idx = [1; cut_points(:); size(trigger_pairs, 1) + 1];

        block_starts = starts(block_boundaries_idx(1:end-1));
        block_ends   = ends_(block_boundaries_idx(2:end) - 1);

        block_bounds_samples = [block_starts, block_ends];
    end

    stim_ext = struct( ...
        'active_channels', active_channels(:), ...
        'trigger_pairs', trigger_pairs, ...
        'block_bounds_samples', block_bounds_samples, ...
        'pulse_sizes', pulse_sizes(:) ...
    );
end