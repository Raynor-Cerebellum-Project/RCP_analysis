function segments = channel_segmentation(stim_channels, params)
chanMap = params.chanMap;
rhs_name = params.name;
dist = params.dist;
sat_thresh = params.sat_thresh;
neuropixel_index = params.neuropixel_index;


if isempty(chanMap)
    chanMap = 'ImecPrimateStimRec128_kilosortChanMap.mat';
end
load(chanMap, 'xcoords', 'ycoords');
indices = [1:length(neuropixel_index)]';

[~, indices_stim] = ismember(stim_channels, neuropixel_index);
if sat_thresh ~= 0
    non_sat_channels = find(max(abs(amplifier_data), [], 2)<=sat_thresh);
else
    non_sat_channels = indices;
end
[~, indices_non_sat] = ismember(non_sat_channels, neuropixel_index);
if dist ~= 0
    indices_non_stim = find(ycoords > (max(ycoords(indices_stim))+dist) | ycoords < (min(ycoords(indices_stim))-dist));
else    
    indices_non_stim = setdiff(indices_non_sat, indices_stim);
end

indices_neighbor_stim = setdiff(indices, cat(1, indices_stim, indices_non_stim));
stim_chan = neuropixel_index(indices_stim);
non_stim_chan = neuropixel_index(indices_non_stim);
neighbor_stim_chan = neuropixel_index(indices_neighbor_stim);
segments.stim = stim_chan;
segments.neighbor_stim = neighbor_stim_chan;
segments.non_stim = non_stim_chan;
segments.id_stim = indices_stim;
segments.id_non_stim = indices_non_stim;
segments.id_non_sat = indices_non_sat;
segments.id_neighbor_stim = indices_neighbor_stim;


% figure
% scatter(xcoords, ycoords, 'DisplayName', 'Neuropixel Probe Channel Map')
% hold on
% scatter(xcoords(indices_stim), ycoords(indices_stim), 'red', 'filled','DisplayName', 'Stim Channels')
% 
% scatter(xcoords(indices_non_stim), ycoords(indices_non_stim), 'green', 'filled','DisplayName', 'Non Stim Channels')
% 
% if sat_thresh ~= 0
%     scatter(xcoords(indices_non_sat), ycoords(indices_non_sat), 'blue','LineWidth',1.5, 'DisplayName', sprintf('Non Saturated Channels below %i threshold', sat_thresh))
% end
% 
% if dist ~= 0 
%     scatter(xcoords(indices_neighbor_stim), ycoords(indices_neighbor_stim), 'yellow','filled', 'DisplayName', sprintf('Stim Channel Neighbors within %i uM', dist))
% end
% 
% hold off
% legend
% title('Neuropixel Probe Channel Map')
% xlabel('xcoords')
% ylabel('ycoords')
% axis([(min(xcoords)-40*range(xcoords)) (max(xcoords)+40*range(xcoords)) 0 (max(ycoords)+100)]);
% 
% 
% figurename = sprintf(rhs_name + "probe_%i_%i.fig", length(indices_stim), length(indices_non_stim));
% 
% subfolderName = sprintf('Channel_Map_Seg_at_%iuM_%isat_thresh', dist, sat_thresh);
% subfolderDir = fullfile(params.outputfolder, subfolderName);
% if ~exist(subfolderDir, 'dir')
%     mkdir(subfolderDir)
% end
% savepath = fullfile(subfolderDir, figurename);
% savefig(savepath)
% close

