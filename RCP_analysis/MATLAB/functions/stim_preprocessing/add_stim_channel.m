
load('stim_data.mat')
% Step 1: Identify channels that have any non-zero stimulation signal
active_channels = find(any(Stim_data, 2));  % Returns row indices of non-zero channels
disp(active_channels)

% Safety check
if isempty(active_channels)
    error('No active stimulation channels found.');
end

% Step 2: Select the first active channel
chosen_channel = active_channels(1);

% Step 3: Extract stimulation signal for this channel
stim_trace = Stim_data(chosen_channel, :);

load('BL_closed_loop_STIM_003_018_Cal.mat')
% Step 4: Index into the stim_trace using Data.Intan_idx to get aligned samples
aligned_stim = stim_trace(Data.Intan_idx);  % Assumes Intan_idx maps into stim_data's 2nd dim

% Step 5: Add to Data.Neural(:,1)
Data.Neural(:,1) = aligned_stim';

save('BL_closed_loop_STIM_003_018_Cal_stim.mat','Data','-v7.3')