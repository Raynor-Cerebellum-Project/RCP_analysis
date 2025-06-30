function [datafolder, filelist] = readfolder(folder_dir, fileid)
    if folder_dir==""
        folder_dir = uigetdir(pwd,'choose an input data folder');
        
    end
    if fileid == ""
        F = dir(folder_dir);
        F(~ismember({F.name},{'.','..'}),:);
    else
        F = dir(fullfile(folder_dir, fileid));
    end
    F = struct2cell(F); 
    datafolder = folder_dir;
    filelist = F(1,:);
end