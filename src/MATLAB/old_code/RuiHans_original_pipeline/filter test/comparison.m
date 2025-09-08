Data_filtered = load("E:\neuraldata\CRR_002_ERASER\filtered\004-8channels-50ms-200hz-40uA_filtered.mat");
Data_raw = load("E:\neuraldata\Caesar_002\seperate_cells\Kilosort4\Project 1 - Occulomotor Kinematics_CELL_7_kilo_11_good\004-8channels-50ms-200hz-40uA_Neural.mat");
%%
% Z = ZoomPlot([Data_filtered.Data.Neural(:, 78), Data_raw.Data.Neural(:, 78)]);
%% mean
[Data_mean, template_mean] = baseline_template_subtraction(Data_raw.Data, 1);
%% movmean
[Data_movmean, template_movmean] = baseline_template_subtraction(Data_raw.Data, 2);
%% ICA
[Data_ICA, template_ICA] = baseline_template_subtraction(Data_raw.Data, 3);
%% PCA
[Data_PCA, template_PCA] = baseline_template_subtraction(Data_raw.Data, 4);
%%
% Z = ZoomPlot([Data_filtered.Data.Neural(:, 78), Data_raw.Data.Neural(:, 78), Data_baseline(:, 78)]);
%% visualize artifact removal

% first extract segments
TRIGDAT =Data_raw.Data.Neural(:, 131);

trigs1 = find(diff(TRIGDAT) < 0); 
trigs2 = find(diff(TRIGDAT) > 0);
if length(trigs1) > length(trigs2)
    trigs  = trigs1;
else
    trigs = trigs2;
end
trigs = trigs(1:2:end);
period = trigs(2) - trigs(1);

NSTIM = length(trigs);

segments_aligned = [];
time_diffs = diff(trigs);
repeat_gap_threshold = period*2;
repeat_boundaries = [0; find(time_diffs > repeat_gap_threshold); numel(trigs)];
num_repeats = numel(repeat_boundaries) - 1;
num_pulse = NSTIM/num_repeats;


for i = 1:NSTIM
    segment = (1 + trigs(i) ):(period+ trigs(i)); 
    segments_aligned = [segments_aligned; segment];  
end
stim_chans = Data_raw.Data.stim_channels;

sample_chans = [stim_chans(1:2), 78, 94, 107];
sample_trials = 1:3:num_repeats;

template_ERASER = zeros(length(sample_trials), num_pulse*period, length(sample_chans));
raw_signal_segs = template_ERASER;
template_mean_tensor = template_ERASER;
template_movmean_tensor = template_ERASER;
template_PCA_tensor = template_ERASER;
template_ICA_tensor = template_ERASER;
ERASER_tensor = template_ERASER;
mean_tensor = template_ERASER;
movmean_tensor = template_ERASER;
PCA_tensor = template_ERASER;
ICA_tensor = template_ERASER;

for i = 1:length(sample_trials)
    sample_trial = sample_trials(i);
    for j = 1:length(sample_chans)
        sample_chan = sample_chans(j);
        sample_pulses = (1+(sample_trial-1)*num_pulse:sample_trial*num_pulse);
        segment = reshape(segments_aligned(sample_pulses, :)', 1, []);
        raw_signal_segs(i, :, j) = Data_raw.Data.Neural(segment, sample_chan);
        template_ERASER(i, :, j) = Data_raw.Data.Neural(segment, sample_chan) - Data_filtered.Data.Neural(segment, sample_chan);
        template_mean_tensor(i, :, j) = Data_raw.Data.Neural(segment, sample_chan) - Data_mean(segment, sample_chan);
        template_movmean_tensor(i, :, j) = Data_raw.Data.Neural(segment, sample_chan) - Data_movmean(segment, sample_chan);
        template_PCA_tensor(i, :, j) = Data_raw.Data.Neural(segment, sample_chan) - Data_PCA(segment, sample_chan);
        template_ICA_tensor(i, :, j) = Data_raw.Data.Neural(segment, sample_chan) - Data_ICA(segment, sample_chan);
        ERASER_tensor(i, :, j) = Data_filtered.Data.Neural(segment, sample_chan);
        mean_tensor(i, :, j) = Data_mean(segment, sample_chan);
        movmean_tensor(i, :, j) = Data_movmean(segment, sample_chan);
        PCA_tensor(i, :, j) = Data_PCA(segment, sample_chan);
        ICA_tensor(i, :, j) = Data_ICA(segment, sample_chan);
    end

end



figure('Name','Eraser Template Vs. Raw')
tiledlayout(length(sample_trials), length(sample_chans))
for i = 1:length(sample_trials)
     sample_trial = sample_trials(i);
    for j = 1:length(sample_chans)
        sample_chan = sample_chans(j);
        nexttile;
        title(sprintf('Channel # %i, Trial # %i',sample_chan, sample_trial ))
        hold on
        for k = 1:period:num_pulse*period
            plot(1:period, raw_signal_segs(i, k:k+period-1, j), 'LineWidth',2.0, 'Color','b')
            
        end
        for k = 1:period:num_pulse*period
            plot(1:period, template_ERASER(i, k:k+period-1, j), 'LineWidth',0.5, 'Color','r')
            
        end
        hold off
        box off
    end
end

figure('Name','Mean Template Vs. Raw')
tiledlayout(length(sample_trials), length(sample_chans))
for i = 1:length(sample_trials)
     sample_trial = sample_trials(i);
    for j = 1:length(sample_chans)
        sample_chan = sample_chans(j);
        nexttile;
        title(sprintf('Channel # %i, Trial # %i',sample_chan, sample_trial ))
        hold on
        for k = 1:period:num_pulse*period
            plot(1:period, raw_signal_segs(i, k:k+period-1, j), 'LineWidth',2.0, 'Color','b')
            
        end
        for k = 1:period:num_pulse*period
            plot(1:period, template_mean_tensor(i, k:k+period-1, j), 'LineWidth',0.5, 'Color','r')
            
        end
        hold off
        box off
    end
end


figure('Name','MovMean Template Vs. Raw')
tiledlayout(length(sample_trials), length(sample_chans))
for i = 1:length(sample_trials)
     sample_trial = sample_trials(i);
    for j = 1:length(sample_chans)
        sample_chan = sample_chans(j);
        nexttile;
        title(sprintf('Channel # %i, Trial # %i',sample_chan, sample_trial ))
        hold on
        for k = 1:period:num_pulse*period
            plot(1:period, raw_signal_segs(i, k:k+period-1, j), 'LineWidth',2.0, 'Color','b')
            
        end
        for k = 1:period:num_pulse*period
            plot(1:period, template_movmean_tensor(i, k:k+period-1, j), 'LineWidth',0.5, 'Color','r')
            
        end
        hold off
        box off
    end
end

figure('Name','ICA Template Vs. Raw')
tiledlayout(length(sample_trials), length(sample_chans))
for i = 1:length(sample_trials)
     sample_trial = sample_trials(i);
    for j = 1:length(sample_chans)
        sample_chan = sample_chans(j);
        nexttile;
        title(sprintf('Channel # %i, Trial # %i',sample_chan, sample_trial ))
        hold on
        for k = 1:period:num_pulse*period
            plot(1:period, raw_signal_segs(i, k:k+period-1, j), 'LineWidth',2.0, 'Color','b')
            
        end
        for k = 1:period:num_pulse*period
            plot(1:period, template_ICA_tensor(i, k:k+period-1, j), 'LineWidth',0.5, 'Color','r')
            
        end
        hold off
        box off
    end
end

figure('Name','PCA Template Vs. Raw')
tiledlayout(length(sample_trials), length(sample_chans))
for i = 1:length(sample_trials)
     sample_trial = sample_trials(i);
    for j = 1:length(sample_chans)
        sample_chan = sample_chans(j);
        nexttile;
        title(sprintf('Channel # %i, Trial # %i',sample_chan, sample_trial ))
        hold on
        for k = 1:period:num_pulse*period
            plot(1:period, raw_signal_segs(i, k:k+period-1, j), 'LineWidth',2.0, 'Color','b')
            
        end
        for k = 1:period:num_pulse*period
            plot(1:period, template_PCA_tensor(i, k:k+period-1, j), 'LineWidth',0.5, 'Color','r')
            
        end
        hold off
        box off
    end
end

figure('Name','Artifact Removed')
tiledlayout(length(sample_trials), length(sample_chans))
for i = 1:length(sample_trials)
     sample_trial = sample_trials(i);
    for j = 1:length(sample_chans)
        sample_chan = sample_chans(j);
        nexttile;
        title(sprintf('Channel # %i, Trial # %i',sample_chan, sample_trial ))
        hold on
        for k = 1:period:num_pulse*period
            plot(1:period, PCA_tensor(i, k:k+period-1, j), 'LineWidth',2.0, 'Color','b')
            
        end
        for k = 1:period:num_pulse*period
            plot(1:period, ERASER_tensor(i, k:k+period-1, j), 'LineWidth',0.5, 'Color','r')
            
        end
        hold off
        box off
    end
end


%%
z=ZoomPlot([Data_filtered.Data.Neural(:, 78), Data_raw.Data.Neural(:,78), Data_PCA(:, 78)])

%%
