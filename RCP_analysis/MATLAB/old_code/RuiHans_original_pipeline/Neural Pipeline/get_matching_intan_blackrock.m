clear all
close all
clc
%% read ccf files
% Get the current directory
currentDirectory = pwd;
ccfdir  = uigetdir('\\10.16.59.34\cullenlab_server\Current Project Databases - NHP\2021 Abducens Stimulation (Neuropixel)\Data\Project 1 - Occulomotor Kinematics\', 'choose root folder for blackrock files');
intandir  = uigetdir('\\10.16.59.34\cullenlab_server\Current Project Databases - NHP\2021 Abducens Stimulation (Neuropixel)\Data\Project 1 - Occulomotor Kinematics\', 'choose root folder for intan files');
datafolder = uigetdir(pwd,'choose an output data folder');
% Define the output Excel file path
%%
prompt = {'Copy the sorted .rhs files to a separate folder (Y/N) : '};
dlgtitle = 'Input';
fieldsize = [1 45];
definput = {'Y'};
flag = inputdlg(prompt,dlgtitle,fieldsize,definput);
%%
outputFile = fullfile(datafolder, 'CCF_Files.xlsx');

% Search for '.ccf' files in the current directory and subfolders
files = dir(fullfile(ccfdir, '**', '*.ccf'));

% Initialize cell array to store file information
data = cell(length(files), 3);

for i = 1:length(files)
    % Extract relevant information
    fullPath = fullfile(files(i).folder, files(i).name);
    
    data{i, 1} = files(i).name;
    data{i, 2} = fullPath;
    data{i, 3} = files(i).date;
end

% Convert to table
T = cell2table(data, 'VariableNames', {'FileName', 'FullPath', 'LastModified'});

% Write to Excel file
writetable(T, outputFile, 'Sheet', 'CCF Files');

disp(['Export complete. The file is saved at ', outputFile]);

%% read setting files
% Search for 'settings.xml' files in the current directory and subfolders
files = dir(fullfile(intandir, '**', 'settings.xml'));

% Initialize cell array to store file information
data = cell(length(files), 4);

for i = 1:length(files)
    % Extract relevant information
    fullPath = fullfile(files(i).folder, files(i).name);
    subfolderName = strsplit(files(i).folder, filesep);
    subfolderName = subfolderName{end};
    
    data{i, 1} = subfolderName;
    data{i, 2} = files(i).name;
    data{i, 3} = fullPath;
    data{i, 4} = files(i).date;
end

% Convert to table
T = cell2table(data, 'VariableNames', {'SubfolderName', 'FileName', 'FullPath', 'LastModified'});

% Write to Excel file
outputFile = fullfile(datafolder, 'Output.xlsx');
writetable(T, outputFile, 'Sheet', 'Setting Files');

disp('Export complete. The file is saved as Output.xlsx in the current directory.');

%%
cd(datafolder)
CCFFiles = readtable('CCF_Files.xlsx');
Output = readtable('Output.xlsx');
%%
datesA = datetime(CCFFiles.LastModified, 'Format','HH:mm:ss');  % Including the time
datesB = datetime(Output.LastModified, 'Format','HH:mm:ss');  % Including the time
% Convert datetime to numeric (in minutes since a reference time)
refTime = min([datesA; datesB]);  % Use the earliest datetime as reference
numA = minutes(datesA - refTime);
numB = minutes(datesB - refTime);

% Define tolerance 
temp = max(abs([numA(:);numB(:)]));

tolerance =0.5/temp;

[isMatch, idxB] = ismembertol(numA, numB, tolerance);  % Check if times in A match with B
%%
matchesA = datesA(isMatch);
matchesB = datesB(idxB(isMatch));

%%
filesA = CCFFiles.FileName(isMatch);
filesB = Output.SubfolderName(idxB(idxB~=0));

T = table(filesA, filesB);
filename = 'matches.xlsx';
writetable(T,filename,'Sheet','matched')
%%

%%
outputfolder = fullfile(datafolder, 'Intan_Sorted');
if ~exist( outputfolder, 'dir')
    mkdir(outputfolder)
end
%%
cd(outputfolder)
for i =  1: length(filesB)
    file = filesB{i};
    folder_path = fullfile(intandir, file, '*.rhs');
    [~,file_name,~] = fileparts(filesA{i});
    disp(['copying .rhs from BR folder ' file_name '...'])
    if ~exist(file_name, 'file')
        mkdir(file_name);
        copyfile(folder_path, file_name)
    end
    
end