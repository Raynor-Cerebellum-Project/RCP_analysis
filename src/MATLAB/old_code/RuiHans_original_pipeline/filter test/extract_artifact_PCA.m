
function artifact_across_chan = extract_artifact_PCA(pulse_chan_data,  stim_chans, nPCs, chanMap)
    
    
    if isempty(chanMap)
        chanMap = 'ImecPrimateStimRec128_kilosortChanMap.mat';
    end
    load(chanMap, 'xcoords', 'ycoords');
    points = [xcoords, ycoords];
    artifact_across_chan = pulse_chan_data;

    [coeff, score, latent] = pca(pulse_chan_data);

    % Keep components explaining 99% variance
    explainedVariance = cumsum(latent) / sum(latent);
    numComponents = find(explainedVariance > 0.80, 1);
    
    % Reduce and reconstruct
    X_reduced = score(:, 1: numComponents) * coeff(:, 1: numComponents)';
    % template = template(1:period_avg+prebuffer, :)'; 
        % S(:,artifactidx) = 0;
        % figure
        % subplot(3,1,1)
        % plot(S(:,artifactidx)*W(1, artifactidx)')
        % title('weighted')
        % subplot(3,1,2)
        % plot(S(:,artifactidx))
        % title('source')
        % subplot(3,1,3)
        % plot(S*W')
        % title('mixed')
        % figure
        % plot(pulse_chan_data(1:300, 78))
        % hold on
        % plot(X_reduced(1:300, 78))
        % hold off
        % 
        % pcs = coeff(:, numComponents+1:128);
        % npc1 = 127-numComponents;
        % npc2 = 128-numComponents;
        % X = coeff(:,npc1:npc2 );
        % nclusters = 16;
        % [idx,C] = kmeans(pcs, nclusters);
        % x1 = min(X(:,1)):0.01:max(X(:,1));
        % x2 = min(X(:,2)):0.01:max(X(:,2));
        % [x1G,x2G] = meshgrid(x1,x2);
        % XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot
        % 
        % idx2Region = kmeans(XGrid,nclusters,'MaxIter',10, 'Start', C(:, npc1:npc2));
        % figure;
        % gscatter(XGrid(:,1),XGrid(:,2),idx2Region,...
        % [0,0.75,0.75;0.75,0,0.75;0.75,0.75,0],'..');
        % hold on;
        % plot(X(:,1),X(:,2),'k*','MarkerSize',5);
        % title 'Fisher''s Iris Data';
        % xlabel 'Petal Lengths (cm)';
        % ylabel 'Petal Widths (cm)'; 
        % legend('Region 1','Region 2','Region 3','Data','Location','SouthEast');
        % hold off;
        % non_stim_chans = setdiff(1:128, stim_chans);

    %     figure
    %     scatter(coeff(stim_chans, end-1), coeff(stim_chans, end-2), 'yellow')
    %     hold on
    %     scatter(coeff(non_stim_chans, end-1), coeff(non_stim_chans, end-2))
    %     xlabel(['PC ' num2str(npc1)]);
    %     ylabel(['PC ' num2str(npc2)]);
    % 
    %     figure;
    %     temp = 123:128 ;
    % for i = 1:length(temp) % Check first few PCs
    %     subplot(length(temp),1,i);
    %     plot(coeff(:,temp(i)));
    %     title(['PC ' num2str(temp(i))]);
    % end
        
        artifact_across_chan = X_reduced;
        % Get the transformation matrix (weights)
    
end

