%This scripts takes the outputs from Kilosort 3.0 (spike sorting) and Phy2 (curation) and stitches unit activity
%back into .mat files that contain kinematic data during our protocols (e.g., translaional acceration signals etc.)
%In order to be properly aligned with the analog signals, the intan indexes must be added to .mat files first via 
%the script "Add_Intan_Index.m"

%Instructions to use: hit run and select all the .mat files (in the Renamed folder) that you wish to stitch unit activity in.
%The output of this script will be saved .mat files containing kinematic data and unit activity in the "separate cells" folder

addpath("\\10.16.59.34\cullenlab_server\Current-Members\Robyn\rFN & Cerebellum\Nodulus_Uvula\Code\stitch_KS25_Chenhao")
addpath(genpath("Z:\___The ANALYSIS Code\Analysis_16APR2021_2pm_backup"))

clc
clear all

save_neural = 0; % <---set to 1 if you want to save raw neural channels with spikes from a cluster

[file_names, Path_name] = uigetfile('.mat', 'MultiSelect', 'on'); %select files in "Renamed folder that have been segmented
[filepath,~,~] = fileparts(Path_name);
[filepath,~,~] = fileparts(filepath);
[~,trackname,~] = fileparts(filepath);

FR_thr = 10;

if ~iscell(file_names)
    file_names = {file_names};
end

xc = [43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27;43;11;59;27];
yc = [20;20;40;40;60;60;80;80;100;100;120;120;140;140;160;160;180;180;200;200;220;220;240;240;260;260;280;280;300;300;320;320;340;340;360;360;380;380;400;400;420;420;440;440;460;460;480;480;500;500;520;520;540;540;560;560;580;580;600;600;620;620;640;640;660;660;680;680;700;700;720;720;740;740;760;760;780;780;800;800;820;820;840;840;860;860;880;880;900;900;920;920;940;940;960;960;980;980;1000;1000;1020;1020;1040;1040;1060;1060;1080;1080;1100;1100;1120;1120;1140;1140;1160;1160;1180;1180;1200;1200;1220;1220;1240;1240;1260;1260;1280;1280];


% load([Path_name '..\KiloSort - Copy\rez.mat'])
load([filepath '\Total (Neural)\rez.mat'])
% rez = get_rez_xy(rez,[filepath '\Total (Neural)\pc_features.npy']); %<-uncomment if using kilo2.5 output, also change to appropriate directories
[~,IDX] = min(abs(rez.xy(:,2)'-xc)); %note these xy coordinates are not normally output from kilsort2.5, which makes this script not run unless 2.5 output is modified or these are changed
[~,IDY] = min(abs(rez.xy(:,1)'-yc)); %note, reverse back x and y for kilo2p5 2 and 1 for kilo3
CH = mod(IDX+1,2)'+IDY';
ST1 = rez.st3(:,1);
SC1 = rez.st3(:,2);
cluster_numbers1 = unique(SC1);

% ST = double(readNPY([Path_name '..\KiloSort\spike_times.npy']));
ST = double(readNPY([filepath '\Total (Neural)\spike_times.npy']));
[~,I1] = sort(ST1);
[~,I2] = sort(ST);
[~,I3] = sort(I2);
I = I1(I3);
CH = CH(I);
SC = double(readNPY([filepath '\Total (Neural)\spike_clusters.npy']));
SC = SC+1;
cluster_numbers = unique(SC);
% [data, ~, raw] = tsvread([Path_name '..\KiloSort\cluster_info.tsv'] );
[data, ~, raw] = tsvread([filepath '\Total (Neural)\cluster_info.tsv'] );
% CH = zeros(size(SC));

% FR = raw(2:end,8); %RLM added
% FR = str2double(FR); %RLM added
FR = zeros(size(cluster_numbers));
for cl_index = cluster_numbers'-1
    idx = find(data(:,1)==cl_index);
    %     ch = data(idx,6);
    %     CH(SC==cl_index) = ch+1;
    
    fr = data(idx,8);
    FR(find(cl_index == cluster_numbers)) = fr;
    
end


N_clusters = length(cluster_numbers);

%%%Manually indexing clusters (so far the only method I am confident with),having trouble properly indexing "good" cells
cluster_numbers = [222 30 121];

quality = raw(2:end,9); % column 4 is KS label, column 9 is group label (manual?)
% quality = raw(2:end,9); %9 is the column that is relabeled
% good_cell_index = (FR>0 & strcmp(quality,'good')); 
good_cell_index = strcmp(quality,'good');
% cluster_numbers = cluster_numbers(good_cell_index)'; %*manually indicating cluster number instead
quality = quality(good_cell_index);
FR = FR(good_cell_index);

cd ../

for file_index = 1:length(file_names)
    
    file_name = file_names{file_index};
    disp(file_name)
    
    
    load([Path_name '\' file_name],'Data'); %load data structure from a .mat file
    Data_back = Data;
    for cell_index = 1:length(cluster_numbers)
        cluster_number = cluster_numbers(cell_index)+1;
        Data = Data_back;
        
        idx = find(SC==cluster_number);
        CH2 = CH(idx);
        CH2 = CH2(~isnan(CH2));
        clusterSites = unique(CH2);
        mainclusterSite = mode(CH2);
        % disp([trackname '_CELL_' num2str(mainclusterSite) '_kilo_'  num2str(cluster_number-1) '_' quality{cell_index}])
        
        N = hist(CH(idx),1:128);
        N = N(clusterSites);
        [~,I] = sort(N,'descend');
        Data.cluster_sites = clusterSites(I);
        Data.Neural_channels = Data.cluster_sites;
        
        spktimes = zeros(length(Data.Intan_idx),1);
        idx = ST(idx);
        [~,idx,~] = intersect(Data.Intan_idx,idx);
        spktimes(idx) = 1;
        
        
        newname = 'ua';
        
        Data.(['spktimes_' newname]) = spktimes;
        Data.(newname) = sign(sum(reshape(spktimes,[30, Data.N])))';
        Data.ChannelList(end+1)={newname};
        try
            Data.ChannelNames(end+1,:)='                ';
            Data.ChannelNames(end,1:length(newname)) = newname;
            Data.ChannelNumbers(end+1)=0;
        catch ERR
            disp(ERR)
        end
        
        Data.adfreq(end+1)=1000; %this is just other info that needs to be added to work with the analysis gui
        Data.samples(end+1)=Data.samples(1);
        Data.SampleCounts=Data.samples;
        Data.NumberOfChannels=length(Data.ChannelList);
        Data.NumberOfSignals=length(Data.ChannelList);
        Data.Definitions(Data.NumberOfSignals)={['Data.' newname '(1+lat:N)']};
        
        fr = fr_estimate(Data.(newname),'kaiser',5,1000);
        
        newname = 'fr';
        Data.(newname) = fr;
        Data.ChannelList(end+1)={newname};
        try
            Data.ChannelNames(end+1,:)='                ';
            Data.ChannelNames(end,1:length(newname)) = newname;
            Data.ChannelNumbers(end+1)=0;
        catch ERR
            disp(ERR)
        end
        
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
        mkdir([Path_name '\..\Seperate cells\Kilosort2p5_Pitch_static\' trackname '_CELL_' num2str(mainclusterSite) '_kilo_'  num2str(cluster_number-1)  '_no_quality'])
        warning on
        
        save([Path_name '\..\Seperate cells\Kilosort2p5_Pitch_static\' trackname '_CELL_' num2str(mainclusterSite) '_kilo_'  num2str(cluster_number-1)  '_no_quality', '\' file_name],'Data','-v7.3');
        
        
    end
end
