% Create delay vector
totalNumTrials = 12;

delay_vector = [ ...
    zeros(1, totalNumTrials / 4), ...
    500 * ones(1, totalNumTrials / 4), ...
    1000 * ones(1, totalNumTrials / 4), ...
    NaN(1, totalNumTrials / 4) ...
];

% Shuffle the vector
delay_vector = delay_vector(randperm(totalNumTrials));

% Define save path
save_path = '/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/BL_RW_003_Session_1/delay_vector.mat';

% Save the variable
save(save_path, 'delay_vector');