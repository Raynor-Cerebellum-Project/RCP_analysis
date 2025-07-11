function [trigs, repeat_boundaries, STIM_CHANS, template_params] = extract_triggers_and_repeats(stim_data, fs, template_params)
% Extracts stimulation triggers, calculates repeat boundaries,
% and updates NSTIM and related parameters.

% === Detect stim channels ===
STIM_CHANS = find(any(stim_data ~= 0, 2));
if isempty(STIM_CHANS)
    warning('No stim signal detected.');
    trigs = [];
    repeat_boundaries = [];
    return;
end

% === Trigger detection ===
TRIGDAT = stim_data(STIM_CHANS(1), :)';
trigs1 = find(diff(TRIGDAT) < 0);
trigs2 = find(diff(TRIGDAT) > 0);
trigs_rz = arrayfun(@(idx) ...
    idx + find(TRIGDAT(idx+1:end) == 0, 1, 'first'), ...
    trigs1, 'UniformOutput', false);
trigs_rz = cell2mat(trigs_rz(~cellfun('isempty', trigs_rz)));

trigs_beg = trigs1;
if length(trigs2) > length(trigs1)
    trigs_beg = trigs2;
end
trigs_beg = trigs_beg(1:2:end);
trigs_end = trigs_rz(2:2:end);

n_trigs = min(length(trigs_beg), length(trigs_end));
trigs = [trigs_beg(1:n_trigs), trigs_end(1:n_trigs)];

% === Update NSTIM ===
template_params.NSTIM = size(trigs, 1);

% === Identify trial groups ===
time_diffs = diff(trigs(:,1));
repeat_gap_threshold = 2 * (2 * template_params.buffer + 1);
repeat_boundaries = [0; find(time_diffs > repeat_gap_threshold); size(trigs, 1)];

end
