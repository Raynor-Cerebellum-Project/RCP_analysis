function [base_root, code_root, base_folder] = set_paths_cullen_lab(session)
% SET_PATHS_CULLEN_LAB
% Determines base paths dynamically from the location of this script.
% Requires the project to have a consistent folder layout.

% Get the directory where this function is stored
this_file_path = mfilename('fullpath');
code_root = fileparts(fileparts((this_file_path)));
base_root = fileparts(fileparts(fileparts(fileparts(fileparts(fileparts(this_file_path))))));
base_folder = fullfile(base_root, session);

end
