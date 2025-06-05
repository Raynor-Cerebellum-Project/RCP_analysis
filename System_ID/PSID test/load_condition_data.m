function [spike_rates_mat_clean, bsv, hsv, t_vec, condition_name] = load_condition_data(base_path, condition_files, condition_idx, cluster_numbers)

    file_name = [condition_files{condition_idx} '.mat'];
    file_path = fullfile(base_path, file_name);
    load(file_path);  % Loads 'Data'

    condition_name = condition_files{condition_idx};

    behavior_hz = 1000;
    bsv = Data.bsv;
    hsv = Data.hsv;
    t_vec = linspace(0, length(bsv)/behavior_hz, length(bsv));

    % Extract neural firing rates
    spike_rates_mat = [];
    for i = 1:length(cluster_numbers)
        field_name = ['fr_' num2str(cluster_numbers(i))];
        data = Data.(field_name);
        spike_rates_mat = [spike_rates_mat; data'];
    end

    % Remove dead channels
    dead_neurons = all(spike_rates_mat == 0, 2);
    spike_rates_mat_clean = spike_rates_mat(~dead_neurons, :);
end
