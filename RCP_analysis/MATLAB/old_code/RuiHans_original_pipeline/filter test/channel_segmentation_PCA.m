function segments = channel_segmentation_PCA(stim_channels, params)
chanMap = params.chanMap;
rhs_name = params.name;

sat_thresh = params.sat_thresh;
neuropixel_index = params.neuropixel_index;
n_nearest = 4;

if isempty(chanMap)
    chanMap = 'ImecPrimateStimRec128_kilosortChanMap.mat';
end
load(chanMap, 'xcoords', 'ycoords');
points = [xcoords, ycoords];
[cIdx,cD] = knnsearch(points,points,'K',n_nearest,'Distance','chebychev');


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

