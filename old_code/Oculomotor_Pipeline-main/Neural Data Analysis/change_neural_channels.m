clc
clear all
close all
[datafolder, folderlist] = readfolder("", "*_CELL_*");
h = waitbar(0, 'Processing...'); % Initialize the progress bar
counter = 0;

for i = 1:length(folderlist)
    
    folder_path = fullfile(datafolder, folderlist{i});
    [~, filelist] = readfolder(folder_path, '*_neural.mat');
    disp(['processing ' folderlist{i}])
    
    for j = 1:length(filelist)
        counter = counter+1;
        disp(['processing ' filelist{j}])
        file_path = fullfile(folder_path, filelist{j});
        try 
            m = matfile(file_path, 'Writable', true);
            % Modify a specific field of the struct
            Data = m.Data;
            Data.Neural_channels = [1:128]'; % 
            m.Data = Data;
            disp('changed sucessfully')
        catch ERR
            disp(ERR)
            continue
        end
        waitbar(counter/(length(folderlist)*length(filelist)), h, sprintf('Progress: %d%%', round((counter/(length(folderlist)*length(filelist)))*100)));
        
    end
    
end
disp('finished')