%% initialize
folder_dir = uigetdir(pwd,'choose a root folder for all filtered neural files');
[datafolder, F] = readfolder(folder_dir, "*_filtered.mat");
outputfolder = uigetdir(pwd, "select folder to store the output");
prompt = {'name session .bin file: '};
dlgtitle = 'bin file naming';
fieldsize = [1 45];
definput = {'all_files_'};
temp = inputdlg(prompt,dlgtitle,fieldsize,definput);
temp = cell2mat(temp);
temp(isspace(temp)) = '_';
trial_number = [];
file_indices = [];
%% sort the folder names with respect to the 5th character 
fileNumberlist = [];
for i = 1:length(F)
        % Extract the number from the filename
        filename = F{i};
        fileidx = split(filename, ["_",".", "-"]);
        fileNumber = str2double(fileidx(1));
        fileNumberlist = [fileNumberlist fileNumber];
end

[~, sorted_idx] = sort(fileNumberlist);
F = F(sorted_idx);
%% read files
if ~isempty(trial_number)
for i = 1:length(F)
    % Extract the number from the filename
    filename = F{i};
    fileidx = split(filename, ["-","."]);
    fileNumber = str2double(fileidx(1));
    % Check if the file number is in the selected ranges
    if any(fileNumber == trial_number)
        file_indices = [file_indices, i]; % Add the index to the list
    end
end
else 
    file_indices = 1:length(F);
end
%% save filtered result to bin
fileID = fopen(fullfile(outputfolder, ['all_files_' temp '.bin']),'w');
for trial_index = file_indices
    neural_path = fullfile(datafolder, F{trial_index});
    load(neural_path);
    fwrite(fileID,int16(Data.Neural(:, 1:128)'),'int16');
end
  
fclose(fileID);
