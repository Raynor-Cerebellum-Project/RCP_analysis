%% remove data with saccades, blinking, and off primary positions and subtract the offset

function [filtered, result]= position_filter_neural(data, params) 
    

    prebuffer = params.prebuffer; % 100 prepulse length ms
    postbuffer = params.postbuffer; % 150 postpulse length ms
    
    tol = params.threshs(1);
    pre_thresh = params.threshs(2); % 100 degree/s the eye velocity threshold for secondary filtering and saccades of primate prior to stimulus
    dur_thresh = params.threshs(3); % 800 deg/ms maximum saccades velocity for primates, threshold during stimulus to filter blinking. 
    pre_thresh_rest = params.threshs(4); % 100 deg/s resting eye velocity threshold
    dur_thresh_rest = params.threshs(5); % 100 deg/s minor movement due to stimulus

    ehp_left_3d = data.ehp_left;
    ehp_right_3d = data.ehp_right;
    evp_left_3d = data.evp_left;
    evp_right_3d = data.evp_right;
    fr = data.fr;
    ua = data.ua;
    if isfield(data, 'ehv_left')
        ehv_left_3d = data.ehv_left;
        ehv_right_3d = data.ehv_right;
        evv_left_3d = data.evv_left;
        evv_right_3d = data.evv_right;

    else
        ehv_left_3d = cell(size(ehp_left_3d,1),1);
        ehv_right_3d = cell(size(ehp_left_3d,1),1);
        evv_left_3d = cell(size(ehp_left_3d,1),1);
        evv_right_3d = cell(size(ehp_left_3d,1),1);

        for i = 1:size(evp_right_3d,1)
            ehv_left_3d{i} = gradient(ehp_left_3d{i});
            ehv_right_3d{i} = gradient(ehp_right_3d{i});
            evv_left_3d{i} = gradient(evp_left_3d{i});
            evv_right_3d{i} = gradient(evp_right_3d{i});
        end
    end
    
    
    %% remove data trials that are not in the primary postion or the offset is too large tha tolerance 

    ehp_left_right = cell(size(evp_right_3d,1),2);
    evp_left_right = cell(size(evp_right_3d,1),2);
    primary_left = zeros(size(evp_right_3d, 1),2);
    primary_right = zeros(size(evp_right_3d, 1),2);
    

    for i = 1:size(evp_right_3d,1)
        bias_ehp_left = mean(ehp_left_3d{i}((prebuffer-20):prebuffer));
        bias_ehp_right = mean(ehp_right_3d{i}((prebuffer-20):prebuffer));
        bias_evp_left = mean(evp_left_3d{i}((prebuffer-20):prebuffer));
        bias_evp_right = mean(evp_right_3d{i}((prebuffer-20):prebuffer));
        % Extract the segment from the data
        ehp_left_right{i,1} = ehp_left_3d{i}-bias_ehp_left;
        ehp_left_right{i,2} = ehp_right_3d{i}-bias_ehp_right;
        evp_left_right{i,1} = evp_left_3d{i}-bias_evp_left;
        evp_left_right{i,2} = evp_right_3d{i}-bias_evp_right;
    
        primary_left(i, 1) = ehp_left_3d{i}(prebuffer)-bias_ehp_left*(bias_ehp_left<tol);
        primary_left(i, 2) = evp_left_3d{i}(prebuffer)-bias_evp_left*(bias_evp_left<tol); % store the primary ehp after removing bias
        primary_right(i, 2) = ehp_right_3d{i}(prebuffer)-bias_ehp_right*(bias_ehp_right<tol);
        primary_right(i, 2) = evp_left_3d{i}(prebuffer)-bias_evp_right*(bias_evp_right<tol);
    end
  
    primary_filter = find(abs(primary_left(:,1))<tol & abs(primary_left(:,2))<tol & abs(primary_right(:,1))<tol & abs(primary_right(:,2))<tol );
    % ehp_left_filtered1 = ehp_left_right(primary_filter,1);
    % ehp_right_filtered1 = ehp_left_right(primary_filter,2);
    % evp_left_filtered1 = evp_left_right(primary_filter,1);
    % evp_right_filtered1 = evp_left_right(primary_filter,2);
    % 
    % ehp_left_filtered1 = ehp_left_right(primary_filter,1);
    % ehp_right_filtered1 = ehp_left_right(primary_filter,2);
    % evp_left_filtered1 = evp_left_right(primary_filter,1);
    % evp_right_filtered1 = evp_left_right(primary_filter,2);
    
    temp = 1:size(evp_right_3d,1);
    removed = setdiff(temp, primary_filter);
    filtered.ehp_left = ehp_left_right(removed,1);
    filtered.ehp_right = ehp_left_right(removed,2);
    filtered.evp_left = evp_left_right(removed,1);
    filtered.evp_right = evp_left_right(removed,1);
    filtered.ehv_left = ehv_left_3d(removed);
    filtered.ehv_right = ehv_right_3d(removed);
    filtered.evv_left = evv_left_3d(removed);
    filtered.evv_right = evv_right_3d(removed);
    filtered.ua = ua(removed);
    filtered.fr = fr(removed);
    filtered.list = removed;

    result.ehp_left = ehp_left_right(primary_filter,1);
    result.ehp_right = ehp_left_right(primary_filter,2);
    result.evp_left = evp_left_right(primary_filter,1);
    result.evp_right =  evp_left_right(primary_filter,2);
    result.ehv_left = ehv_left_3d(primary_filter);
    result.ehv_right = ehv_right_3d(primary_filter);
    result.evv_left = evv_left_3d(primary_filter);
    result.evv_right = evv_right_3d(primary_filter);
    result.ua = ua(primary_filter);
    result.fr = fr(primary_filter);
    result.list = primary_filter;

end