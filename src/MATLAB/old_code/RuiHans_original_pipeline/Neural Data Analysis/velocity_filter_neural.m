function [filtered, result]= velocity_filter_neural(data, params) 
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
%% filter based on eye velocity
    slope_pre_H_move = zeros(size(ehp_left_3d,1),2); %  eye horizontal velocity prior to stimulus
    slope_pre_rest = zeros(size(ehp_left_3d,1),2); %  rest of the eye velocity prior to stimulus
    slope_dur_H_move = zeros(size(ehp_left_3d,1),2); % eye horizontal velocity during stimulus
    slope_dur_rest = zeros(size(ehp_left_3d,1),2); % rest of the eye velocity during stimulus
    slope_post_H_move = zeros(size(ehp_left_3d,1),2); % eye horizontal velocity after stimulus
    slope_post_rest = zeros(size(ehp_left_3d,1),2); % rest of the eye velocity after stimulus
    for i = 1: size(ehp_right_3d,1) 
        slope_pre_H_move(i, 1) = max(abs(ehv_left_3d{i}(1:prebuffer-2)));
        slope_pre_H_move(i, 2) = max(abs(ehv_right_3d{i}(1:prebuffer-2)));
        slope_pre_rest(i, 1) = max(abs(evv_left_3d{i}(1:prebuffer-2)));
        slope_pre_rest(i, 2) = max(abs(evv_right_3d{i}(1:prebuffer-2)));
        slope_dur_H_move(i, 1) = max(abs(ehv_left_3d{i}(prebuffer:end-postbuffer)));
        slope_dur_H_move(i, 2) = max(abs(ehv_right_3d{i}(prebuffer:end-postbuffer)));
        slope_dur_rest(i, 1) = max(abs(evv_left_3d{i}(prebuffer:end-postbuffer)));
        slope_dur_rest(i, 2) = max(abs(evv_right_3d{i}(prebuffer:end-postbuffer)));
        slope_post_H_move(i, 1) = max(abs(gradient(ehp_left_3d{i}(end-postbuffer:end))));
        slope_post_H_move(i, 2) = max(abs(gradient(ehp_right_3d{i}(end-postbuffer:end))));
        slope_post_rest(i, 1) = max(abs(gradient(evp_left_3d{i}(end-postbuffer:end))));
        slope_post_rest(i, 2) = max(abs(gradient(evp_right_3d{i}(end-postbuffer:end))));
    end

    [filter1, ~] = find(slope_pre_H_move < pre_thresh & slope_dur_H_move < dur_thresh &  slope_post_H_move <0.8);
    [filter2, ~] = find(slope_pre_rest < pre_thresh_rest & slope_dur_rest < dur_thresh_rest & slope_post_rest < 0.8);
    total_filter = [filter1; filter2];
    t = tabulate(total_filter);
    % disp(t)
    % disp(total_filter)
    
    primary_filter = t(t(:,2)==4, 1).';
    % disp(primary_filter)
    temp = 1:size(evp_right_3d,1);
    removed = setdiff(temp, primary_filter);
    
    filtered.ehp_left = ehp_left_3d(removed);
    filtered.ehp_right = ehp_right_3d(removed);
    filtered.evp_left = evp_left_3d(removed);
    filtered.evp_right = evp_right_3d(removed);
    filtered.ehv_left = ehv_left_3d(removed);
    filtered.ehv_right = ehv_right_3d(removed);
    filtered.evv_left = evv_left_3d(removed);
    filtered.evv_right = evv_right_3d(removed);
    filtered.ua = data.ua(removed);
    filtered.fr = data.fr(removed);
    filtered.list = removed;

    result.ehp_left = ehp_left_3d(primary_filter);
    result.ehp_right = ehp_right_3d(primary_filter);
    result.evp_left = evp_left_3d(primary_filter);
    result.evp_right = evp_right_3d(primary_filter);
    result.ehv_left = ehv_left_3d(primary_filter);
    result.ehv_right = ehv_right_3d(primary_filter);
    result.evv_left = evv_left_3d(primary_filter);
    result.evv_right = evv_right_3d(primary_filter);
    result.ua = data.ua(primary_filter);
    result.fr = data.fr(primary_filter);
    
    result.list = primary_filter;
end
