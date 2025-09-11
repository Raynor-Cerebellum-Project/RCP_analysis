folder = uigetdir("E:\neuraldata\Caesar_002_rev3\seperate_cells\Kilosort4");
[datafolder, filelist] = readfolder(folder, '*_neural.mat');

cd(datafolder)
VAFs = zeros(length(filelist), 1);
for i = 1:length(filelist)
    load(filelist{i})
%% Preprocess data
thr = 1;
thr2 = 0.02;% threshold for saccades
X = Data.ehp_left_3d;
X1 = gradient(X);
X_other = Data.ehp_right_3d;
X1_other = gradient(X_other);
% Ensure correct filtering
validIdx = and(X1 < thr, X1_other<thr2, X1 >thr2);
X = X(validIdx);
X1 = X1(validIdx);
y = max(0, Data.fr(validIdx));

% Number of bootstrap samples
nBoot = 1000;

% Define function to compute delayed regression model
fun = @(params, data) compute_error(params, data);

% Initial guesses: [k, r, b, td]
ic = [0, 0, 0, 1]; 

% Regression function for bootstrapping
regFunc = @(data) fminsearch(@(params) fun(params, data), ic);

% Prepare data for bootstrapping
data = [X, X1, y];
coeffMean = regFunc(data);
% %%
% % Perform bootstrap sampling and regression
% bootCoeffs = bootstrp(nBoot, regFunc, data);
% 
% % Compute mean and 95% confidence intervals
% coeffMean = mean(bootCoeffs);
% coeffCI = prctile(bootCoeffs, [2.5, 97.5]);
% 
% % Display results
% fprintf('Intercept: Mean = %.3f, 95%% CI = [%.3f, %.3f]\n', ...
%         coeffMean(3), coeffCI(1,3), coeffCI(2,3));
% fprintf('Slope X: Mean = %.3f, 95%% CI = [%.3f, %.3f]\n', ...
%         coeffMean(1), coeffCI(1,1), coeffCI(2,1));
% fprintf('Slope X1: Mean = %.3f, 95%% CI = [%.3f, %.3f]\n', ...
%         coeffMean(2), coeffCI(1,2), coeffCI(2,2));
% fprintf('Delay (td): Mean = %.3f, 95%% CI = [%.3f, %.3f]\n', ...
%         coeffMean(4), coeffCI(1,4), coeffCI(2,4));
% 
% % Plot bootstrap distributions
% figure;
% subplot(2,2,1);
% histogram(bootCoeffs(:,1), 30);
% title('Bootstrap Distribution of k (Slope X)');
% 
% subplot(2,2,2);
% histogram(bootCoeffs(:,2), 30);
% title('Bootstrap Distribution of r (Slope X1)');
% 
% subplot(2,2,3);
% histogram(bootCoeffs(:,3), 30);
% title('Bootstrap Distribution of Intercept');
% 
% subplot(2,2,4);
% histogram(bootCoeffs(:,4), 30);
% title('Bootstrap Distribution of Delay (td)');

%% Compute Variance Accounted For (VAF)
td_opt = floor(coeffMean(4)); % Use optimized delay
if td_opt > 0 && td_opt < length(X)
    X_shifted = X(1:end-td_opt);
    X1_shifted = X1(1:end-td_opt);
    y_shifted = y(td_opt+1:end);
else
    X_shifted = X;
    X1_shifted = X1;
    y_shifted = y;
end

y_hat = coeffMean(1) * X_shifted + coeffMean(2) * X1_shifted + coeffMean(3);
VAF = (1 - var(y_hat - y_shifted) / var(y_shifted)) * 100;
VAFs(i) = VAF;
disp(filelist{i})
fprintf('VAF = %.2f%%\n', VAF);
end
%% Function to compute error for fminsearch
function error = compute_error(params, data)
    k = params(1);
    r = params(2);
    b = params(3);
    td = floor(params(4)); % Ensure delay is an integer
    
    X = data(:,1);
    X1 = data(:,2);
    y = data(:,3);
    
    % Prevent indexing issues
    if td > 0 && td < length(X)
        X_shifted = X(1:end-td);
        X1_shifted = X1(1:end-td);
        y_shifted = y(td+1:end);
    else
        X_shifted = X;
        X1_shifted = X1;
        y_shifted = y;
    end
    
    % Compute squared error
    error = rms(k * X_shifted + r * X1_shifted + b - y_shifted);
end
