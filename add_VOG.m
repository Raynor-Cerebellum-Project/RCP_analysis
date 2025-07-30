
[Calib_file, Calib_path] = uigetfile('.mat','Select Calibrate file');
[VOG_name, VOG_path] = uigetfile('.txt','Select VOG file');

Calib_name = [Calib_path Calib_file];

filename = [VOG_path VOG_name];
D = readmatrix(filename);
ND = size(D,2);

fileID = fopen(filename,'r');
formatSpec = '%s';
C_text = textscan(fileID,formatSpec,ND,'Delimiter',' ');
channel_names = C_text{1,1};
%%
close all;

load(Calib_name,'Data')
trigger = (Data.Neural(:,1))/11; % Selects column with trigger pulses to match frames and align signals
% idx = find(diff((triger>1500))==1); 
idx = find(diff(trigger)>100);  


figure % Plots torsion trace
hold on
plot(D(:,33))

figure % Plots GVS and indexed frame points
hold on
plot(trigger)
plot(idx,1500*ones(size(idx)),'*')

L = size(D,1); % D is the Open iris data
t_ = idx/30; % downsamples to 1KHz if smapled at 30KHz
t = ceil(t_(1)):floor(t_(end));
 
if L ~= length(idx)
    disp([num2str(L-length(idx)) ' sample(s) dropped']) % Samples will drop if sampling rates are not equal
    t_ = linspace(t_(1),t_(end),L);
end

for ch_index = 1:length(channel_names)
    channel_name = channel_names{ch_index};
    tmp = D(:,ch_index);
% 
    tmp2 = zeros(Data.N,1);
    tmp2(t) = interp1(t_,tmp,t);
%         figure(1)
%         hold off
%         plot(tmp2)
%         hold on
%         plot(tmp)
    Data.(channel_name) = reshape(tmp2,[],1);

    if ~sum(strcmp(Data.ChannelList,channel_name))

        Data.ChannelList(end+1)={channel_name};
        try
            Data.ChannelNames(end+1,:)='                ';
            if length(channel_name)<16
                Data.ChannelNames(end,1:length(channel_name)) = channel_name;
            else
                Data.ChannelNames(end,:) = channel_name(1:16);
            end    
            Data.ChannelNumbers(end+1)=0;
        catch ERR
            disp(ERR)

        end

        Data.adfreq(end+1)=1000;
        Data.samples(end+1)=Data.samples(1);
        Data.SampleCounts=Data.samples;
        Data.NumberOfChannels=length(Data.ChannelList);
        Data.NumberOfSignals=length(Data.ChannelList);
        Data.Definitions(Data.NumberOfSignals)={['Data.' channel_name '(1+lat:N)']};
    end
end

save([Calib_name(1:end-4), 'wVOG.mat'],'Data')
