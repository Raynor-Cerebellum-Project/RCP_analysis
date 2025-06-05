clc
clear all
close all
[datafolder, folderlist] = readfolder("", "");
outputdir = uigetdir(pwd, 'choose a folder to save the output');
p.prebuffer = 100; %prepulse length ms
p.postbuffer = 150; %postpulse length ms
% p.num_pulse_threshold = 5; %An BR file should have >5 pulses
p.fc1 = 125;
p.fc2 = 75;
p.fs = 1000; 
p.flag = 1;
p.threshs = [45, 0.3, 0.6, 0.3, 0.3];
session_name = "Daphne-session-3";
%%
Trialsetting = readtable('003_Experiment_Trial_setting.xlsx');

%%
M_corr = zeros(length(folderlist), 50);
for i = 1:length(folderlist)
    outputfolder = fullfile(outputdir, folderlist{i});
    mkdir(outputfolder)
    folder_path = fullfile(datafolder, folderlist{i});
    [~, filelist] = readfolder(folder_path, '*_neural.mat');
    for j = 1:length(filelist)
        file_path = fullfile(folder_path, filelist{j});
        parts = strsplit(filelist{j}, '-');
        number = parts{1};
        if (number == "047") ||( number == "049")
            continue
        end
        [~, Processed_Data, ~] = pipeline_neural(file_path, p, session_name);
        ua = Processed_Data.ua;
        timeframe = Processed_Data.timeframe;
        if ~isempty(ua)
            fr = mean(cell2mat(Processed_Data.fr), 1);
            ehv = mean(cell2mat(Processed_Data.ehv_left), 1);
            baseline_fr = fr(1:p.prebuffer);
            sigma = std(baseline_fr);
            mu = mean(baseline_fr);
            act_fr = fr(p.prebuffer+1 : end-p.postbuffer);
            z_fr = (act_fr-mu)./sigma;
            M_corr(i, str2double(number)) = min(corrcoef(z_fr, ehv(p.prebuffer+1 : end-p.postbuffer)), [], "all");
        end
    end
end

%%
T_corr = table(folderlist', M_corr);
% T_corr.Properties.VariableNames = {num2str([10:17, 40, 43:45])};
T_corr.Variables = T_corr.Variables(2:end, :);

%%
filename = 'correlation_matrix.xlsx';
writetable(T_corr,filename,'Sheet','correlation_matrix_reg')