%% Setup
clear; close all; clc;
addpath(genpath(fullfile('..', 'functions')));

session = 'BL_RW_003_Session_1';
[base_root, code_root, base_folder] = set_paths_cullen_lab(session);

intan_folder = fullfile(base_folder, 'Intan');
trial_dirs = dir(fullfile(intan_folder, 'BL_closed_loop_STIM_*'));

fs = 30000;
fixed_params = struct( ...
    'NSTIM', 0, ...
    'buffer', 25, ...
    'template_leeway', 15, ...
    'stim_neural_delay', 13, ...
    'movmean_window', 3, ...
    'med_filt_range', 25, ...
    'gauss_filt_range', 25 ...
    );
template_modes = {'local_drift_corr'};

% Spike detection + FR params
filt_range = [300 6000];
[b, a] = butter(3, filt_range / (fs/2), 'bandpass');
rate_mode = 'kaiser';
cutoff_freq = 5;
threshold_std = 3;
refractory_ms = 1;
target_fs = 1000;
ds_factor = round(fs / target_fs);

%% Select trial 5
trial_id = 5;
trial = trial_dirs(trial_id).name;
trial_path = fullfile(intan_folder, trial);
fprintf('\nProcessing trial: %s\n', trial);

% === Load stim data ===
stim_struct = load(fullfile(trial_path, 'stim_data.mat'));
if isfield(stim_struct, 'Stim_data')
    stim_data = stim_struct.Stim_data;
elseif isfield(stim_struct, 'stim_data')
    stim_data = stim_struct.stim_data;
else
    error('No stim_data found.');
end

[trigs, repeat_boundaries, STIM_CHANS, updated_params] = ...
    extract_triggers_and_repeats(stim_data, fs, fixed_params);

if isempty(trigs)
    error('No triggers found.');
end

neural_struct = load(fullfile(trial_path, 'neural_data.mat'));
neural_data = neural_struct.neural_data;
%%
plot_chans = [1, 21, 45, 46, 48, 49, 65, 66, 69, 96];

spike_trains_all = cell(numel(plot_chans), 1);
smoothed_fr_all  = cell(numel(plot_chans), 1);

for i = 1:numel(plot_chans)
    ch = plot_chans(i);
    fprintf('Artifact correction for channel %d...\n', ch);

    [cleaned_versions, block2]  = ...
        compare_template_modes(neural_data, updated_params, template_modes, ...
        trigs, repeat_boundaries, trial_path, ch);

    fprintf('Spike detection for channel %d...\n', ch);
    raw = double(cleaned_versions(ch, :));  % Get just the channel you asked for
    filtered = filtfilt(b, a, raw);

    med = median(filtered);
    mad_val = median(abs(filtered' - med'));
    robust_std = 1.4826 * mad_val;
    thresh = threshold_std * robust_std;

    spike_idx = find(abs(filtered - med) > abs(thresh));
    isi = diff(spike_idx);
    spike_idx = spike_idx([true, isi > fs * refractory_ms / 1000]);

    spike_train = zeros(length(filtered), 1);
    spike_train(spike_idx) = 1;

    if exist('block2', 'var') && ~isempty(block2) && isfield(block2, 'range_end')
        t = (block2.full_range_start:block2.range_end) - block2.full_range_start;
        t = t / fs * 1000;  % ms
        final_trace = double(cleaned_trace(block2.full_range_start:block2.range_end));

        fig = figure('Name', 'Block 2: Drift + Template Removal', 'Color', 'w');
        plot(t, block2.original_trace, 'k-', 'LineWidth', 0.8); hold on;
        plot(t, block2.gauss_filtered, 'r-', 'LineWidth', 0.6);
        plot(t, final_trace, 'b-', 'LineWidth', 1.2);

        for k = 1:length(block2.template_subtractions)
            ts = block2.template_subtractions{k};
            if ~isempty(ts)
                plot(t, ts, 'Color', [0.3 0.7 0.3], 'LineStyle', '--');
            end
        end

        for k = 1:length(block2.interpulse_drifts)
            ds = block2.interpulse_drifts{k};
            if ~isempty(ds)
                plot(t, ds, 'Color', [0.7 0.3 0.7], 'LineStyle', ':');
            end
        end

        legend({'Original', 'Drift (Gauss)', 'Final Cleaned', ...
            'Pulse Templates', 'Interpulse Drifts'}, 'Location', 'best');
        xlabel('Time (ms)');
        ylabel('Amplitude');
        title(sprintf('Overlay: Block 2 | Channel %d', ch));
        grid on;

        % === Save Figure ===
        out_dir = fullfile(trial_path, 'Figures');
        if ~exist(out_dir, 'dir'), mkdir(out_dir); end
        out_name = fullfile(out_dir, sprintf('Block2_Cleaning_Channel_%03d.png', ch));
        exportgraphics(fig, out_name, 'Resolution', 300);
        close(fig);  % optional: close to avoid cluttering GUI
    end

    fr_full = fr_estimate(spike_train, rate_mode, cutoff_freq, fs);
    smoothed_fr_all{i} = downsample(fr_full, ds_factor);
    spike_trains_all{i} = spike_train;
end

