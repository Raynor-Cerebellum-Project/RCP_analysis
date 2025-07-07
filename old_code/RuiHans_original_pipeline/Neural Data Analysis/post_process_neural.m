function output = post_process_neural(processed_data)
alpha = 0.05; % alpha: significance level (e.g., 0.05 for 95% confidence interval)
output =cell(size(processed_data,1), 1) ;
for i = 1: size(processed_data, 1)
    if size(processed_data, 1) > 1    
        temp = processed_data{i};
    else
        temp = processed_data;
    end
    num_bootstrap = size(temp.timeframe, 2);
    
    temp.ehp_ipsi = cell2mat(temp.ehp_left);
    temp.evp_ipsi = cell2mat(temp.evp_left);
    temp.ehp_contra = cell2mat(temp.ehp_right);
    temp.evp_contra = cell2mat(temp.evp_right);
    temp.fr_mat = cell2mat(temp.fr);
    temp.ehv_ipsi = cell2mat(temp.ehv_left);
    temp.evv_ipsi = cell2mat(temp.evv_left);
    temp.ehv_contra = cell2mat(temp.ehv_right);
    temp.evv_contra = cell2mat(temp.evv_right);
    
    if any(structfun(@isempty, temp))
        output{i} = [];
        continue
    end
    % get EH
    [ipsi_ehp_avg, CI_ipsi_ehp_lower, CI_ipsi_ehp_upper] = regular_confidence_interval(temp.ehp_ipsi, alpha, num_bootstrap);
    temp.ipsi_ehp_avg = ipsi_ehp_avg;
    temp.CI_ipsi_ehp_lower = CI_ipsi_ehp_lower;
    temp.CI_ipsi_ehp_upper = CI_ipsi_ehp_upper;
    [ipsi_evp_avg, CI_ipsi_evp_lower, CI_ipsi_evp_upper] = regular_confidence_interval(temp.evp_ipsi, alpha, num_bootstrap);
    temp.ipsi_evp_avg = ipsi_evp_avg;
    temp.CI_ipsi_evp_lower = CI_ipsi_evp_lower;
    temp.CI_ipsi_evp_upper = CI_ipsi_evp_upper;
    [contra_ehp_avg, CI_contra_ehp_lower, CI_contra_ehp_upper] = regular_confidence_interval(temp.ehp_contra, alpha, num_bootstrap);
    temp.contra_ehp_avg = contra_ehp_avg;
    temp.CI_contra_ehp_lower = CI_contra_ehp_lower;
    temp.CI_contra_ehp_upper = CI_contra_ehp_upper;
    [contra_evp_avg, CI_contra_evp_lower, CI_contra_evp_upper] = regular_confidence_interval(temp.evp_contra, alpha, num_bootstrap);
    temp.contra_evp_avg = contra_evp_avg;
    temp.CI_contra_evp_lower = CI_contra_evp_lower;
    temp.CI_contra_evp_upper = CI_contra_evp_upper;
    
    % get EV
    [ipsi_ehv_avg, CI_ipsi_ehv_lower, CI_ipsi_ehv_upper] = regular_confidence_interval(temp.ehv_ipsi, alpha, num_bootstrap);
    temp.ipsi_ehv_avg = ipsi_ehv_avg;
    temp.CI_ipsi_ehv_lower = CI_ipsi_ehv_lower;
    temp.CI_ipsi_ehv_upper = CI_ipsi_ehv_upper;
    [ipsi_evv_avg, CI_ipsi_evv_lower, CI_ipsi_evv_upper] = regular_confidence_interval(temp.evv_ipsi, alpha, num_bootstrap);
    temp.ipsi_evv_avg = ipsi_evv_avg;
    temp.CI_ipsi_evv_lower = CI_ipsi_evv_lower;
    temp.CI_ipsi_evv_upper = CI_ipsi_evv_upper;
    [contra_ehv_avg, CI_contra_ehv_lower, CI_contra_ehv_upper] = regular_confidence_interval(temp.ehv_contra, alpha, num_bootstrap);
    temp.contra_ehv_avg = contra_ehv_avg;
    temp.CI_contra_ehv_lower = CI_contra_ehv_lower;
    temp.CI_contra_ehv_upper = CI_contra_ehv_upper;
    [contra_evv_avg, CI_contra_evv_lower, CI_contra_evv_upper] = regular_confidence_interval(temp.evv_contra, alpha, num_bootstrap);
    temp.contra_evv_avg = contra_evv_avg;
    temp.CI_contra_evv_lower = CI_contra_evv_lower;
    temp.CI_contra_evv_upper = CI_contra_evv_upper;
    

    % get fr
    [temp.fr_avg, temp.CI_fr_lower, temp.CI_fr_upper] = regular_confidence_interval(temp.fr_mat, alpha, num_bootstrap);
    
    output{i} = temp;
end


end


function [mean_data, ci_lower, ci_upper] = regular_confidence_interval(data, alpha, num)
    % data: n*m matrix (n trials, m readings)
    alpha;% confidence_level: e.g., 0.95 for 95% confidence interval
    % num: place holder that do nothing
    
    % Calculate mean across trials (rows)
    mean_data = mean(data, 1);
    
    % Get the number of trials (rows)
    n_trials = size(data, 1);
    
    % Calculate standard error of the mean
    sem = std(data, 0, 1) / sqrt(n_trials); 
    
    % Determine the t-statistic for the desired confidence level
    t_stat = tinv(1-( alpha) / 2, n_trials - 1);
    
    % Calculate the margin of error
    margin_of_error = t_stat * sem;
    
    % Confidence interval lower and upper bounds
    ci_lower = mean_data - margin_of_error;
    ci_upper = mean_data + margin_of_error;
end



function [mean_vals, CI_lower, CI_upper] = bootstrap_confidence_interval(data, alpha, num_bootstrap)
    % Function to calculate bootstrap confidence interval for the mean across trials
    % data: n*m matrix, where n is the number of trials and m is the length of each trial
    % alpha: significance level (e.g., 0.05 for 95% confidence interval)
    % num_bootstrap: number of bootstrap samples (e.g., 1000 or more)
    % Returns the mean values, and the lower and upper bounds of the confidence interval

    % Number of trials (rows) and data length (columns)
    [n, m] = size(data);
   
    % Step 1: Compute the mean across trials for the original dataset
    mean_vals = mean(data, 1);  % Mean along the first dimension (trials)

    % Step 2: Initialize an array to store the bootstrap sample means
    bootstrap_means = zeros(num_bootstrap, m);
   
    % Step 3: Perform bootstrapping
    for i = 1:num_bootstrap
        % Resample the data (with replacement) across the first dimension (trials)
        resampled_data = data(randi(n, [n, 1]), :);
       
        % Compute the mean for this resampled dataset
        bootstrap_means(i, :) = mean(resampled_data, 1);
    end
   
    % Step 4: Compute the confidence interval bounds (percentile method)
    CI_lower = prctile(bootstrap_means, alpha/2 * 100, 1);
    CI_upper = prctile(bootstrap_means, (1 - alpha/2) * 100, 1);
end