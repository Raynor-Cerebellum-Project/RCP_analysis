intan_file = "D:\neuraldata\Caesar_002\Intan_Sorted\CRR_NPXL_STIM_002_007\CRR_NPXL_STIM_002__210507_180800.rhs";
read_Intan_RHS2000_file(intan_file);

%%
chanMap = 'ImecPrimateStimRec128_042421.mat';
load(chanMap, 'xcoords', 'ycoords');
points = [xcoords, ycoords];
[cIdx,cD] = knnsearch(points,points,'K',5,'Distance','chebychev');
neuropixel_index = [    18, 19, 20, 21, 22, 23, 24, 25, ...
   26, 27, 29, 17, 2,  32, 1,  30, ...
    31, 39, 3,  36, 38, 28, 35, 37, ...
    4,  34, 16, 33, 15, 14, 13, 12, ...
    11, 10, 9,  8,  7,  6,  5,  63, ...
    59, 56, 64, 58, 55, 40, 57, 54, ...                                                                                               
    41, 60, 53, 43, 61, 52, 44, 62, ...
    51, 42, 47, 50, 45, 48, 49, 46, ...
    65, 96, 69, 66, 95, 68, 67, 94, ...
    70, 83, 93, 72, 84, 92, 71, 85, ...
    91, 73, 88, 90, 81, 87, 89, 82, ...
    86, 108, 107, 106,105,104,103,102,...
    101,100,99, 98, 80, 97, 79, 109,...
    76, 78, 117,75, 77, 110,74, 114,...
    115,112,113,111,128,116,118,119,...
    120,121,122,123,124,125,126,127];
    neighbors = neuropixel_index(cIdx);


%% extract segments based on trigs
fs = 30000;
STIM_CHANS = find(any(stim_data~=0, 2));
% if ~isempty(STIM_CHANS)
    TRIGDAT = stim_data(STIM_CHANS(1),:)';
  
    i = 1;
    neighbor_channels = amplifier_data( neighbors(i,:), :);
    trigs1 = find(diff(TRIGDAT) < 0); % Biphasic pulse, first phase negative.
    trigs2 = find(diff(TRIGDAT) > 0);
    if length(trigs1) > length(trigs2)
        trigs  = trigs1;
    else
        trigs = trigs2;
    end
    trigs = trigs(1:2:end);
    
    NSTIM = length(trigs);
    period = trigs(2)- trigs(1);
    prebuffer = 0;
    segments = [];
    for i = 1:NSTIM
        segment =  (-prebuffer+1 + trigs(i) ):min(period+ trigs(i), length(amplifier_data) );
        segments = [segments; segment];
    end
    
%% visualize
figure 
plot(ppchannels)
%% variance in the curve to the mean
rmse = zeros(5, size(segments, 1));
for i = 1: size(segments, 1)
    ppchannels = neighbor_channels(:, segments( i, :));
    template = mean(ppchannels,1);
    rmse(:, i) = rms(ppchannels - template,2);
end

%%
figure

h = heatmap(rmse./max(rmse));

%%
[coeff, score, latent] = pca(ppchannels);

% Keep components explaining 99% variance
explainedVariance = cumsum(latent) / sum(latent);
numComponents = find(explainedVariance >0.98, 1);
X_reduced = score(:, numComponents+1:end) * coeff(:, numComponents+1:end)';
template = mean(ppchannels,1);
figure 
plot(1:76, ppchannels)
hold on 
plot(template, 'LineWidth',3)
legend(string(neighbors(1, :)))

% end

