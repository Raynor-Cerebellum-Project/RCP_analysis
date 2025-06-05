function output = ReadBin(filename,numOfchannels,channel_idx,sample_idx)
%%
% Bin filename
% Number of channels Neuropixel 1.0 =384, Read/Write = 128
% Channels of interest
% Samples of interest

fileID = fopen(filename,'r');


output = zeros(length(channel_idx),length(sample_idx));
L_before = sample_idx(1)-1;
L = range(sample_idx)+1;
range_vector = min(sample_idx):max(sample_idx);
NL = floor(L_before/1e6);
for n = 1:NL 
    tmp = fread(fileID,[numOfchannels,1e6],'int16');
end
tmp = fread(fileID,[numOfchannels,mod(L_before,1e6)],'int16');


NL = floor(L/1e6);
for n = 1:NL 
    tmp = fread(fileID,[numOfchannels,1e6],'int16');
    [~,IA,IB] = intersect(range_vector((n-1)*1e6+1:n*1e6),sample_idx);
    output(:,IB) = tmp(channel_idx,IA);
end
tmp = fread(fileID,[numOfchannels,mod(L,1e6)],'int16');
[~,IA,IB] = intersect(range_vector(NL*1e6+1:end),sample_idx);
output(:,IB) = tmp(channel_idx,IA);

output = output';
fclose(fileID);

