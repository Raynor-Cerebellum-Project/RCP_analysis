function [base_root, code_root, base_folder] = set_paths_cullen_lab(session)

this_file_path = mfilename('fullpath');
code_root = fileparts(fileparts(this_file_path));

[~, hostname] = system('hostname');
hostname = strtrim(hostname);

if contains(hostname, {'navon', 'navona', 'gaon', 'foo'})
    base_root = '/cis/net/io109/data/RCPstorage';
else
    % raynor-pc: traverse up from code location
    base_root = fileparts(fileparts(fileparts(fileparts(fileparts(fileparts(this_file_path))))));
end

base_folder = fullfile(base_root, session);

end