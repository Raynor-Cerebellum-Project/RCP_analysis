function [removed, remained, error] = pipeline_neural(file_dir, params, session_name)
sample = load(file_dir);
[~, file_name, ~] = fileparts(file_dir);
description = str2double(regexp(file_name, '\d+', 'match'));

number = description(1);
density = description(2);
dur = description(3); % stimulation duration in ms
freq = description(4); % stimulation frequency in Hz 
current = description(5); % stimulation current in uA

%% parameters for preprocessing
% p.prebuffer = 100; %prepulse length ms
% p.postbuffer = 150; %postpulse length ms
% % p.num_pulse_threshold = 5; %An BR file should have >5 pulses
% p.fc1 = 250;
% p.fc2 = 125;
% p.fs = 1000; 
% 
% p.threshs = [20, 0.1, 0.8, 0.1, 0.1];
% session_name = "Daphne session 3";
p = params;
%% extract trials from original data 
p.fc = params.fc1;
[D0,timeframe] = extract_segments_with_neural(sample, p);

%% freq filter first and then do artifact removal
% p.threshs = [30, 0.3, 0.8, 0.3, 0.3];
% p.fc1 = 250;
% p.fc2 = 75;
% p.flag = 1; % to use the frequency filter on velocity only
D1 = freq_filter_neural(D0, p);
[D1_removed, D1_remained] = velocity_filter_neural(D1, p);
[D2_removed, D2_remained] = position_filter_neural(D1_remained, p);
% D3_removed = freq_filter(D2_removed, p);
% D3_remained = freq_filter(D2_remained, p);


p.flag = 1;
p.fc = params.fc2;
D3_remained = freq_filter_neural(D2_remained, p);
D3_removed = freq_filter_neural(D2_removed, p);

%% visualization
% processed1 = {D1_remained, D1_removed};
% processed2 = {D2_remained,D2_removed};
% p.filters ={"velocity filter", "velocity filter"};
% visualize(D0, processed1, timeframe, p, file_name, session_name, 0)
% visualize(D0, processed1, timeframe, p,  file_name, session_name, 1)
% 
% p.filters ={"position filter", "position filter"};
% visualize(D1_remained, processed2, timeframe, p, file_name, session_name, 0)
% visualize(D1_remained, processed2, timeframe, p,  file_name, session_name, 1)




%% return parameters
% error = compare_data(D0, processed1, processed2);
error = 0;
removed = D3_removed;
removed.timeframe = timeframe;
remained= D3_remained;
remained.timeframe = timeframe;
end
