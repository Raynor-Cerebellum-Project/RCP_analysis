%This scripts takes the outputs from Kilosort 3.0 (spike sorting) and Phy2 (curation) and stitches unit activity
%back into .mat files that contain kinematic data during our protocols (e.g., translaional acceration signals etc.)
%In order to be properly aligned with the analog signals, the intan indexes must be added to .mat files first via 
%the script "Add_Intan_Index.m"

%Instructions to use: hit run and select all the .mat files (in the Renamed folder) that you wish to stitch unit activity in.
%The output of this script will be saved .mat files containing kinematic data and unit activity in the "separate cells" folder

clc
clear all

save_neural = 0; % <---set to 1 if you want to save raw neural channels with spikes from a cluster

% [file_names, Path_name] = uigetfile('.mat', 'MultiSelect', 'on'); %select files in "Renamed folder that have been segmented

% Path_name = 'E:\neuraldata\Daphne_003_mat\Renamed-Copy\';
% F = dir(fullfile(Path_name, '*_neural.mat'));
% F = struct2cell(F);
% F = F(1,:);
% mat_files = uigetdir('\\10.16.59.34\cullenlab_server\Current Project Databases - NHP\2021 Abducens Stimulation (Neuropixel)\Data\Project 1 - Occulomotor Kinematics\', 'choose root folder for _neural.mat files to stitch kilosort 4 result');
mat_files = uigetdir(pwd, 'choose root folder for _neural.mat files to stitch kilosort 4 result');

[Path_name, F] = readfolder(mat_files, '*_neural.mat');

file_path= uigetdir(pwd, "select folder for kilosort4 results files");
%%
[bin_file, location] = uigetfile('*.bin', "select .bin file that store the artifact removed neural data");
%%
outputfolder = uigetdir(pwd, "select output folder");
outputfolder = fullfile(outputfolder, 'seperate_cells');
% Neural_back = ReadBin(bin_file,128);
if ~exist(outputfolder, 'file')
    mkdir(outputfolder)
end
file_indices = [];
% trial_number =[10:17, 19, 32:38, 40, 43:45];
% trial_number = [10:17, 40, 43:45];
% trigger_file_path = 'D:\Oculomotor Research\Current_non-currtent\Neural data analysis\bin_test\mid_bot_all_session_trigger\';
% trial_number = [4, 8, 14, 20];
trial_number = [];
file_num_list = [];
%% sort the folder names with respect to the 5th character 
fileNumberlist = [];
for i = 1:length(F)
        % Extract the number from the filename
        filename = F{i};
        fileidx = split(filename, ["-","_","."]);
        fileNumber = str2double(fileidx(1));
        fileNumberlist = [fileNumberlist fileNumber];
end

[~, sorted_idx] = sort(fileNumberlist);
F = F(sorted_idx);
%%
for i = 1:length(F)
    % Extract the number from the filename
    filename = F{i};
    fileidx = split(filename, ["-","_","."]);
    fileNumber = str2double(fileidx(1));
    % Check if the file number is in the selected ranges
    if ~isempty(trial_number)
        if any(fileNumber == trial_number)
            file_indices = [file_indices, i]; % Add the index to the list
            file_num_list = [file_num_list, fileNumber];
        end
    else 
        file_indices = [file_indices, i];
        file_num_list = [file_num_list, fileNumber];
    end
end
file_names = F(file_indices);



[filepath,~,~] = fileparts(Path_name);
[filepath,~,~] = fileparts(filepath);
[~,trackname,~] = fileparts(filepath);


%file_path = 'E:\kilosort_result\allfile_test_mid_bot_003_no2021\kilosort4\';
FR_thr = 15;

% if ~iscell(file_names)
%     file_names = {file_names};
% end

xc = [43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27];
yc = [20;20;40;40;60;60;80;80;100;100;120;120;140;140;160;160;180;180;200;200;220;220;240;240;260;260;280;280;300;300;320;320;340;340;360;360;380;380;400;400;420;420;440;440;460;460;480;480;500;500;520;520;540;540;560;560;580;580;600;600;620;620;640;640;660;660;680;680;700;700;720;720;740;740;760;760;780;780;800;800;820;820;840;840;860;860;880;880;900;900;920;920;940;940;960;960;980;980;1000;1000;1020;1020;1040;1040;1060;1060;1080;1080;1100;1100;1120;1120;1140;1140;1160;1160;1180;1180;1200;1200;1220;1220;1240;1240;1260;1260;1280;1280];
% rez = get_rez_xy(rez,"E:\kilosort_result\all_file_test\378\pc_features.npy"); %<-uncomment if using kilo2.5 output, also change to appropriate directories
% % load([Path_name '..\KiloSort - Copy\rez.mat'])
% % load([filepath '\Sorting results\KiloSort2p5\rez.mat'])
% % rez = get_rez_xy(rez,[filepath '\Sorting results\KiloSort2p5\pc_features.npy']); %<-uncomment if using kilo2.5 output, also change to appropriate directories
xy = double(readNPY([file_path '\spike_positions.npy']));
[~,IDX] = min(abs(xy(:,1)'-xc)); %note these xy coordinates are not normally output from kilsort2.5, which makes this script not run unless 2.5 output is modified or these are changed
[~,IDY] = min(abs(xy(:,2)'-yc)); %note, reverse back x and y for kilo2p5 2 and 1 for kilo3
CH = mod(IDX+1,2)'+IDY'; % channel map connected
% ST1 = rez.st3(:,1); % spike time in sample

% CH = double(readNPY([file_path 'channel_map.npy']));
SC1 = double(readNPY([file_path '\spike_templates.npy']));
% SC1 = rez.st3(:,2); % spike clusters
cluster_numbers1 = unique(SC1);

% ST = double(readNPY([Path_name '..\KiloSort\spike_times.npy']));
% ST1 = rez.st3(:,1)
ST = double(readNPY([file_path '\spike_times.npy']));
% [~,I1] = sort(ST1);
[~,I2] = sort(ST);
[~,I3] = sort(I2);
%%
I = I2(I3);
%%
CH = CH(I);
% SC = double(readNPY([file_path 'spike_clusters.npy']));
SC = SC1+1;
cluster_numbers = unique(SC);
[data, ~, raw] = tsvread([file_path '\cluster_info.tsv'] );
% CH = zeros(size(SC));
%%
% FR = raw(2:end,8); %RLM added
% FR = str2double(FR); %RLM added
cluster_numbers = unique(SC);
FR = zeros(size(cluster_numbers));

% for cl_index = cluster_numbers'-1
%     idx = find(data(:,1)==cl_index);
%     %     ch = data(idx,6);
%     %     CH(SC==cl_index) = ch+1;
% 
%     fr = data(idx,8);
%     FR(find(cl_index == cluster_numbers)) = fr;
% 
% end
FR = data(2:end, 8);
quality = raw(2:end,9);

% quality = raw(2:end,9); %9 is the column that is relabeled
good_cell_index = (FR>FR_thr & strcmp(quality,'good'));
%%
% good_cell_index = (strcmp(quality,'good'));
cluster_numbers = cluster_numbers(good_cell_index)'; %*manually indicating cluster number instead
quality = quality(good_cell_index);
FR = FR(good_cell_index);



N_clusters = length(cluster_numbers);

%%%Manually indexing clusters (so far the only method I am confident with),having trouble properly indexing "good" cells

%NPXL 10 Session 1
% cluster_numbers = [25]% 25 40 41 42 45 47 48 50 54 71 76 91 105 108 135 137 140 148 152 153 154 155 157 159 160 161 164]; 010 Session 1? 
%NPXL 006 Session 2
% cluster_numbers = [2 113 162 115 85 118 137 122 141 126 127 130 150 152 169 59 172 174 182]; 
%NPXL 006 Session 2 Faisals
%cluster_numbers = [123 114 92 120 125 129 131 71 62 133 94 40 24 25 97 23 69 85 138 98 108 2];
%NPXL 011 Session 1
%cluster_numbers = [177 162 62 178 13 167 60 189 121 193 176 61 119 182 59 175 41 67 173 174 108 25 185 187 188 10]% 194 197 12 91 27 133 45 140];
%cluster_numbers = [194 197 12 91 27 133 45 140];
%NPXL 011 Session 1 Fiasals 
%cluster_numbers = [99 103 102 115 105 107 42 2 120 109 36 111]; % 78 116 113 121 74 112 29 118 22 73 32 25 27 11 58];% 16 9 84 33 34 21 44 17 122];
% cluster_numbers = [27 11 58 16 9 84 33 34 21 44 17 122];
%NPXL 004 Session 2
% cluster_numbers = [36 103 131 92 35 133 137 116 99 132]% 40 18 138 13 139 134 141 135 26 140 128 1 104 39 3 30 110 115]
%NPXL_002_Session_4
%cluster_numbers = [1 17 6 7 15];

%%
% cd ../
h = waitbar(0, 'Processing...'); % Initialize the progress bar
segment_mark = 0; % 0 for the first neural data segment
counter = 0;
for file_index = 1:length(file_names)
    file_name = file_names{file_index};
    disp(file_name)
    
    % disp('segment_mark')
    % disp(segment_mark)

    load([Path_name '\' file_name],'Data'); %load data structure from a .mat file
    Data_back = Data;
    % extract segements from sc, ch, by operating on ST
    Data_NStruct = cell(1,length(cluster_numbers));
    start_intan = Data.Intan_idx(1);

    for cell_index = 1:length(cluster_numbers)
        counter = counter+1;
        cluster_number = cluster_numbers(cell_index);
        Data = Data_back;
        
        idx = find(SC==cluster_number);
        CH2 = CH(idx);
        CH2 = CH2(~isnan(CH2));
        clusterSites = unique(CH2);
        mainclusterSite = mode(CH2);
        disp([trackname '_CELL_' num2str(mainclusterSite) '_kilo_'  num2str(cluster_number-1) '_' quality{cell_index}])
        
        N = hist(CH(idx),1:128);
        N = N(clusterSites);
        [~,I] = sort(N,'descend');
        Data.cluster_sites = clusterSites(I);
        sample =Data.Intan_idx-start_intan+segment_mark;
        Data.Neural = Data_back.Neural(:, Data.cluster_sites);
        Data.Neural_channels = [Data.cluster_sites; 1; 2];

        artifact_removed = ReadBin([location bin_file],128,Data.cluster_sites(1), sample);
        
        preprocessed = ReadBin([file_path '\temp_wh.dat'],128,Data.cluster_sites(1), sample);

        
       
        Data.Neural = [Data.Neural, artifact_removed, preprocessed];
        
        spktimes = zeros(length(Data.Intan_idx),1);
        idx = ST(idx)-segment_mark;
        idx = idx(idx>=0);
        % disp('segment_index')
        % disp(length(idx))
        
        [~,idx,~] = intersect(Data.Intan_idx-start_intan,idx);
        % disp('index_intersect')
        % disp(length(idx))
        spktimes(idx) = 1;
        
        
        newname = 'ua';
        
        Data.(['spktimes_' newname]) = spktimes;
        Data.(newname) = sign(sum(reshape(spktimes,[30, Data.N])))';
        if ~any(strcmp(Data.ChannelList, newname))
            Data.ChannelList(end+1)={newname};
            try
                Data.ChannelNames(end+1,:)='                ';
                Data.ChannelNames(end,1:length(newname)) = newname;
                Data.ChannelNumbers(end+1)=0;
            catch ERR
                disp(ERR)
            end
        end
        
        
        Data.adfreq(end+1)=1000; %this is just other info that needs to be added to work with the analysis gui
        Data.samples(end+1)=Data.samples(1);
        Data.SampleCounts=Data.samples;
        Data.NumberOfChannels=length(Data.ChannelList);
        Data.NumberOfSignals=length(Data.ChannelList);
        Data.Definitions(Data.NumberOfSignals)={['Data.' newname '(1+lat:N)']};
        
        fr = fr_estimate(Data.(newname),'kaiser',10,1000); %lower cut-off 2*50+1
        
        newname = 'fr';
        Data.(newname) = fr;
        if ~any(strcmp(Data.ChannelList, newname))
            Data.ChannelList(end+1)={newname};
            try
                Data.ChannelNames(end+1,:)='                ';
                Data.ChannelNames(end,1:length(newname)) = newname;
                Data.ChannelNumbers(end+1)=0;
            catch ERR
                disp(ERR)
            end
        end

        % sample =Data.Intan_idx+segment_mark;
        % newname = 'preprocessed';
        % Data.(newname) = preprocessed_neural(sample, Data.Neural_channels(1));
        % if ~any(strcmp(Data.ChannelList, newname))
        %     Data.ChannelList(end+1)={newname};
        %     try
        %         Data.ChannelNames(end+1,:)='                ';
        %         Data.ChannelNames(end,1:length(newname)) = newname;
        %         Data.ChannelNumbers(end+1)=0;
        %     catch ERR
        %         disp(ERR)
        %     end
        % end
        % newname = 'artifact_removed';
        % % Data.(newname) = artifact_removed(sample, Data.Neural_channels(1));
        % % Data.(newname) = ReadBin([location bin_file],128,Data.cluster_sites(1),sample);
        % if ~any(strcmp(Data.ChannelList, newname))
        %     Data.ChannelList(end+1)={newname};
        %     try
        %         Data.ChannelNames(end+1,:)='                ';
        %         Data.ChannelNames(end,1:length(newname)) = newname;
        %         Data.ChannelNumbers(end+1)=0;
        %     catch ERR
        %         disp(ERR)
        %     end
        % end

        Data.adfreq(end+1)=1000;
        Data.samples(end+1)=Data.samples(1);
        Data.SampleCounts=Data.samples;
        Data.NumberOfChannels=length(Data.ChannelList);
        Data.NumberOfSignals=length(Data.ChannelList);
        Data.Definitions(Data.NumberOfSignals)={['Data.' newname '(1+lat:N)']};
        
        
        if save_neural == 1 %adds raw neural data from channels containing a cluster
            Data.Neural = ReadBin([Path_name '\' strrep(file_name,'.mat','.bin')],128,Data.cluster_sites,1:30*Data.N);
%             Data.Neural = ReadBin([Path_name '\..\Intan\all_files.bin'],128,Data.cluster_sites,Data.Intan_idx);
        end
        
        warning off
        mkdir([outputfolder '\Kilosort4\' trackname '_CELL_' num2str(mainclusterSite) '_kilo_'  num2str(cluster_number-1)  '_' quality{cell_index}])
        warning on 

        save([outputfolder '\Kilosort4\' trackname '_CELL_' num2str(mainclusterSite) '_kilo_'  num2str(cluster_number-1)  '_' quality{cell_index} '\' file_name],'Data','-v7.3');
        waitbar(counter/(length(file_names)*length(cluster_numbers)), h, sprintf('Prosessing %s %d%%', file_name, round(cell_index/length(cluster_numbers)*100)));

        % Data_NStruct{cell_index} = Data;
    end
    segment_mark = segment_mark+length(Data.Intan_idx);
    
 
    % warning off
    % mkdir([Path_name '\..\NStruct_cells\Kilosort4\' trackname])
    % warning on
    % save([Path_name '\..\NStruct_cells\Kilosort4\' trackname '\' file_name],'Data_NStruct','-v7.3');

end
