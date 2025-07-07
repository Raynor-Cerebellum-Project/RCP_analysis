%% parameters for preprocessing
p.prebuffer = 100; %prepulse length ms
p.postbuffer = 150; %postpulse length ms
% p.num_pulse_threshold = 5; %An BR file should have >5 pulses
p.fc1 = 250;
p.fc2 = 125;
p.fs = 1000; 

p.threshs = [20, 0.1, 0.8, 0.1, 0.1];
session_name = "Daphne-session-3";

%% test the distrotion of different frequency on the data under different pipeline structure
% n_trials = 5;
% freq_range = linspace(50, 125, n_trials);
% error_tot = zeros(2, n_trials);
% Filelist = dir('*uA.mat');
% 
% for j = 1:n_trials
%     error = zeros(size(Filelist,1),2);
%     p.fc2 = freq_range(j);
%     for i=1:size(Filelist,1)
%         [~, ~, error(i, :)] = pipeline(Filelist(i).name, p, session_name);
%     end
%     error(isnan(error)) = 0;
%     error_tot(:,j) = median(error).';
% end
% 
% %% visualize metric
% figure 
% plot(freq_range, error_tot(1, :), '-*','DisplayName', 'freq filter first')
% hold on 
% plot(freq_range, error_tot(2, :), '--+', 'DisplayName', 'freq filter last')
% hold off
% legend 
% xlabel('cutoff frequency (Hz)')
% ylabel('metric')
% title('SNR at different Cutoff Frequency with two pipeline')

%%

