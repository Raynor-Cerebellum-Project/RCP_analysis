function [base_root, code_root, base_folder] = set_paths_cullen_lab(session)
% SET_PATHS_CULLEN_LAB
% Determines base paths dynamically from the location of this script.
% Requires the project to have a consistent folder layout.

% Get the directory where this function is stored
this_file_path = mfilename('fullpath');
base_root = fileparts(fileparts(fileparts(this_file_path)));  % Go up two levels

% Construct relative paths
code_root = fullfile(base_root, 'Analysis Codes');
base_folder = fullfile(base_root, 'Data', session);

end
