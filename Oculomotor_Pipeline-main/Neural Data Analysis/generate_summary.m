[datafolder, folderlist] = readfolder("", "*_CELL_*");
outputdir = uigetdir(datafolder, 'choose a folder to save the output');
[~, filelist] = readfolder(fullfile(datafolder, folderlist{1}), "*_neural.mat");
filename = 'summary.xlsx';
T_summary = table(folderlist');
% T_summary.Properties.VariableNames = filelist;
writetable(T_summary,fullfile(outputdir, filename),'Sheet','summary')