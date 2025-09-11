 clear all
 close all
%% initialize
folder_dir = uigetdir(pwd,'choose a root folder for all sorted intan files');
[datafolder, F] = readfolder(folder_dir, "*_STIM_*");
outputfolder = uigetdir(pwd, "select folder to store the output");
prompt = {'name session .bin file: '};
dlgtitle = 'bin file naming';
fieldsize = [1 45];
definput = {'all_files_'};
temp = inputdlg(prompt,dlgtitle,fieldsize,definput);
temp = cell2mat(temp);
temp(isspace(temp)) = '_';
%%
neuropixel_index = [    18, 19, 20, 21, 22, 23, 24, 25, ...
   26, 27, 29, 17, 2,  32, 1,  30, ...
    31, 39, 3,  36, 38, 28, 35, 37, ...
    4,  34, 16, 33, 15, 14, 13, 12, ...
    11, 10, 9,  8,  7,  6,  5,  63, ...
    59, 56, 64, 58, 55, 40, 57, 54, ...                                                                                               
    41, 60, 53, 43, 61, 52, 44, 62, ...
    51, 42, 47, 50, 45, 48, 49, 46, ...
    65, 96, 69, 66, 95, 68, 67, 94, ...
    70, 83, 93, 72, 84, 92, 71, 85, ...
    91, 73, 88, 90, 81, 87, 89, 82, ...
    86, 108, 107, 106,105,104,103,102,...
    101,100,99, 98, 80, 97, 79, 109,...
    76, 78, 117,75, 77, 110,74, 114,...
    115,112,113,111,128,116,118,119,...
    120,121,122,123,124,125,126,127];
% neuropixel_index = [    18, 19, 20, 21, 22, 23, 24, 25, ...
%     26, 27, 29, 17, 2,  32, 1,  30, ... 31, 55, 3,  52, 54, 28, 51, 53,
%     ... 4,  50, 16, 49, 15, 14, 13, 12, ... 11, 10, 9,  8,  7,  6,  5,
%     47, ... 43, 40, 48, 42, 39, 56, 41, 38, ... 57, 44, 37, 59, 45, 36,
%     60, 46, ... 35, 58, 63, 34, 61, 64, 33, 62, ... 81, 80, 85, 82, 79,
%     84, 83, 78, ... 86, 67, 77, 88, 68, 76, 87, 69, ... 75, 89, 72, 74,
%     65, 71, 73, 66, ... 70, 108, 107, 106, 105,104,103,102,...
%     101,100,99, 98, 96, 97, 95, 109,... 92, 94, 117,91, 93, 110,90,
%     114,... 115,112,113,111,128,116,118,119,...
%     120,121,122,123,124,125,126,127];


probe_params = struct('dist', 0, ... % the largest distance between stim channel neighbors and stim channels
    'chanMap', 'ImecPrimateStimRec128_042421.mat', ... % the channel mapping we are using should be a .mat file this is also used for kilosort4
    'sat_thresh', 0, ... % The maximum value of amplifier data before saturation, 0 for not segmenting the channels based on saturation
    'neuropixel_index', neuropixel_index, ... % the channel indexing for neuropixel. 
    'name', '', ... % name the rhs file name
    'outputfolder', outputfolder ...
    );

template_params = struct( 'NSTIM', 0, ...  % number of stim pulses
    'isstim', true, ... % true if the data is from a stim channel
    'period_avg', 30, ... % number of points to average for the template
    'start', 40, ... % skip the first number of pulses when calculating the template
    'buffer', 0, ... % thenumber of points before each oulse to be considered in calculating the template
    'skip_n', 0 ...% number of initial pulses to skip to calculate the template
    );
visualize = ""; % if we need to visualize the result 
% "stim": visualize only result of stim channels
% "neighbor-stim": visualize the stim channels and its neighboring
% channels
% "non-stim": only non-stim channels
% "all": all the channels
% "": none
%%


session_trigger = [];
% trial_number =[10:17, 19, 32:38, 40, 43:45];
% trial_number =[10:17, 40, 43:45];
trial_number = [1, 7, 13, 19];
% trial_number = [4, 8, 14, 20];
file_indices = [];
% trial_number = [];



fileID = fopen(fullfile(outputfolder, ['all_files_' temp '.bin']),'w');

session_trigger_folder =fullfile(outputfolder, [temp '_session_trigger']);
if ~exist(session_trigger_folder, 'dir')
    mkdir(session_trigger_folder)
end
%% sort the folder names with respect to the 5th character 
fileNumberlist = [];
for i = 1:length(F)
        % Extract the number from the filename
        filename = F{i};
        fileidx = split(filename, ["_","."]);
        fileNumber = str2double(fileidx(5));
        fileNumberlist = [fileNumberlist fileNumber];
end

[~, sorted_idx] = sort(fileNumberlist);
F = F(sorted_idx);
%%
if ~isempty(trial_number)
    for i = 1:length(F)
        % Extract the number from the filename
        filename = F{i};
        fileidx = split(filename, ["_","."]);
        fileNumber = str2double(fileidx(5));
        % Check if the file number is in the selected ranges
        if any(fileNumber == trial_number)
            file_indices = [file_indices, i]; % Add the index to the list
        end
    end
else 
    file_indices = 1:length(F);
end
%%


for trial_index = file_indices
    intan_path = fullfile(datafolder, F{trial_index});
    fileidx = split(F{trial_index}, ["_","."]);
    fileNumber = str2double(fileidx(5));
    % intan = dir(intan_path);
    % 
    % intan = struct2cell(intan);
    % intan = intan(1,3:end);
    % intan = intan(contains(intan,'.rhs'));
    [~, intan] = readfolder(intan_path, "*.rhs");
    intan_files = sort(intan);


for intan_file_index = 1:length(intan_files)
    
    intan_file = fullfile(intan_path, intan_files{intan_file_index});
    [~,name,~] = fileparts(intan_file);
    probe_params.name = name;
    template_params.name = name;
    disp(['Processing intant files from trial: '   num2str(fileNumber) ': ' num2str(name)])
    read_Intan_RHS2000_file(intan_file);
  
   
    
    % try
%     amplifier_data_copy = amplifier_data;
    amplifier_data_copy = artifact_Removal(amplifier_data, stim_data, probe_params, template_params, visualize);
    % catch 
    %     continue
    % end

    fwrite(fileID,int16(amplifier_data_copy(probe_params.neuropixel_index,:)),'int16');
    session_trigger = [session_trigger board_adc_data(1:2,:)];
    
    clearvars -except session_trigger fileID probe_params template_params  visualize_channels intan_file_index intan_path F intan_files trial_index datafolder visualize session_trigger_folder fileNumber
    
end
    
    session_trigger_file = sprintf('session_trigger_%i.mat', fileNumber);
    savepath = fullfile(session_trigger_folder, session_trigger_file);
    save(savepath,'session_trigger','-v7.3')
    session_trigger = [];
end

fclose(fileID);
% save('session_trigger.mat','session_trigger','-v7.3')

%%
% read_Intan_RHS2000_file("E:\neuraldata\Daphne_003\DRL_NXPL_STIM_003_004.ccf\DRL_NXPL_STIM_003__220427_152339.rhs");
% 
% 
%  %%
%  n = length(board_adc_data);
% 
%  session_trigger(2, 1:10)
%  session_trigger(2, [1:10]+n)
% 
%  %% 
% A = readNPY("C:\PHY_data\all_file_test\357\spike_times.npy");
% B = readNPY("C:\PHY_data\all_file_test\357\spike_templates.npy");
% C = readNPY("C:\PHY_data\all_file_test\357\spike_positions.npy");
% D = readNPY("C:\PHY_data\all_file_test\357\spike_clusters.npy");