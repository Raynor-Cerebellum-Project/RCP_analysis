% Set working directory and filename
data_dir = '/Volumes/CullenLab_Server/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/Data/BL_RW_003_Session_1/Calibrated/IntanFile_10';
cd(data_dir);
filename = 'BL_closed_loop_STIM_003_012_Cal.mat';

% Load file
load(filename, 'Data');

% Get original fields
old_fields = fieldnames(Data.segments);

% Create new structure
new_segments = struct();

for i = 1:numel(old_fields)
    f = old_fields{i};

    % Match suffix pattern
    tokens = regexp(f, '^(.*?)(_nan|_0|_100|_200)(_neg|_pos)?$', 'tokens');
    if ~isempty(tokens)
        base = tokens{1}{1};
        suffix = tokens{1}{2};
        polarity = '';
        if numel(tokens{1}) > 2
            polarity = tokens{1}{3};
        end
        new_field = [base polarity suffix];
    else
        new_field = f;
    end

    new_segments.(new_field) = Data.segments.(f);
end

% Replace and save
Data.segments = new_segments;
save(filename, 'Data', '-v7.3');

fprintf('Renamed and saved segment fields in: %s\n', filename);
