function base_root = set_paths()
% Return the correct CullenLab server root path based on host machine.

host = char(getHostName(java.net.InetAddress.getLocalHost));

switch host
    case 'raynor-pc'
        base_root = '/mnt/cullen';
    case 'Bryans-MacBook.local'
        base_root = '/Volumes/CullenLab_Server';
    otherwise
        warning('Unknown host: %s. Please choose root folder manually.', host);
        selected = uigetdir(pwd, 'Select root folder "Ex: CullenLab_Server"');
        if isequal(selected, 0)
            error('No folder selected. Exiting.');
        end
        base_root = selected;
end
end
