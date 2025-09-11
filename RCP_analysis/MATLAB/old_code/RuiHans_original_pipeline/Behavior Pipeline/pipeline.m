function [removed, remained, error] = pipeline(file_dir, params, session_name)
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
[D0,timeframe] = extract_segments(sample, p);


% %% freq filter first and then do artifact removal
% D1 = freq_filter(D0, p);
% [D2_removed, D2_remained] = artifact_filter(D1, p);
% processed1 = {D1, D2_remained, D2_removed};
% p.filters = {"frequency filter", "artifact filter"};
% % visualize(D0, processed1, timeframe, p, file_name, session_name, 0)
% % visualize(D0, processed1, timeframe, p, file_name, session_name, 1)

%% artifact removal first and then do freq filter
[D1_removed, D1_remained] = artifact_filter(D0, p);
D2_removed = freq_filter(D1_removed, p);
D2_remained = freq_filter(D1_remained, p);
processed2 = {D1_remained, D2_remained, D2_removed};
p.filters = {"artifact filter", "frequency filter" };
% visualize(D0, processed2, timeframe, p, file_name, session_name, 0)
% visualize(D0, processed1, timeframe, p,  file_name, session_name, 1)

%% return parameters
% error = compare_data(D0, processed1, processed2);
error = 0;
removed = processed2{3};
remained= processed2{2};
end
