%% Ask ONCE (outside loop)
nChannelsNpxl = questdlg("How many channels are in the neural data?", ...
    "nChannelsNpxl", "64", "128", "other", "other");

if (nChannelsNpxl == "128") 
    neuropixel_index = [18,19,20,21,22,23,24,25,...
        26,27,29,17,2,32,1,30,...
        31,39,3,36,38,28,35,37,...
        4,34,16,33,15,14,13,12,...
        11,10,9,8,7,6,5,63,...
        59,56,64,58,55,40,57,54,...
        41,60,53,43,61,52,44,62,...
        51,42,47,50,45,48,49,46,...
        65,96,69,66,95,68,67,94,...
        70,83,93,72,84,92,71,85,...
        91,73,88,90,81,87,89,82,...
        86,108,107,106,105,104,103,102,...
        101,100,99,98,80,97,79,109,...
        76,78,117,75,77,110,74,114,...
        115,112,113,111,128,116,118,119,...
        120,121,122,123,124,125,126,127];
elseif (nChannelsNpxl == "64")
    neuropixel_index = 1:64;
else
    error("no channel number entered");
end

%% Get all subfolders
parent_dir = pwd;
D = dir(parent_dir);
D = D([D.isdir]);                         % only folders
D = D(~ismember({D.name},{'.','..'}));    % remove . and ..
D = D(1:7); %**** specify files----------------------------------------------

%% Loop through each folder
for i_folder = 1:length(D)
    
    folder_name = D(i_folder).name;
    folder_path = fullfile(parent_dir, folder_name);
    
    disp(['Processing folder: ' folder_name])
    cd(folder_path)
    
    %% -------- YOUR ORIGINAL CODE STARTS HERE --------
    
    F = dir();
    F = struct2cell(F);
    F = F(1,3:end);
    F = F(contains(F,'.rhs'));
    intan_files = sort(F);

    session_triger = [];
    fileID = fopen('all_files.bin','w');

    clear Data
    for i = neuropixel_index
        Data.ChannelList{i} = strcat('Ch',num2str(i));
        Data.(Data.ChannelList{i}) = [];
    end

    neural_data = [];
    Stim_data = [];

    for intan_file_index = 1:length(intan_files)
        intan_file = intan_files{intan_file_index};
        disp(['Processing Intan file: ' intan_file])

        read_Intan_RHS2000_file(intan_file);

        fwrite(fileID, int16(amplifier_data(neuropixel_index,:)), 'int16');

        session_triger = [session_triger board_adc_data(1:2,:)];

        if nChannelsNpxl == "128"
            Stim_data = [Stim_data, stim_data(1:128,:)];

            for ch = 1:128
                analog = amplifier_data(ch,:);
                Data.(sprintf('Ch%d', ch)) = [Data.(sprintf('Ch%d', ch)) analog];
            end

        elseif nChannelsNpxl == "64"
            neural_data = [neural_data, amplifier_data(1:64,:)];

            for ch = neuropixel_index
                analog = amplifier_data(ch,:);
                analog = downsample(analog, 30, 10);
                Data.(sprintf('Ch%d', ch)) = [Data.(sprintf('Ch%d', ch)) analog];
            end
        end
    end

    Data.Neural = neural_data;
    Data.Stim = Stim_data;

    fclose(fileID);

    save('session_triger.mat','session_triger','-v7.3')    
    save('neural_data.mat','neural_data','-v7.3')
    save("stim_data.mat","Stim_data","-v7.3")

    %% -------- YOUR ORIGINAL CODE ENDS HERE --------
    
    cd(parent_dir)   % go back before next loop
end