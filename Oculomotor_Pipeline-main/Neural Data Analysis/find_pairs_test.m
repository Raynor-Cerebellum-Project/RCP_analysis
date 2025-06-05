clear all
experiment_setting = readtable("D:\Oculomotor Research\Experiment Summary\Experiment_Setting_Summary.xlsx", "Sheet","CRR_NXPL_STIM_002");
%%
durs = experiment_setting.Duration_ms_;
freqs = experiment_setting.Frequency_Hz_;
ids = experiment_setting.BR;
locs = experiment_setting.region_of_stim;
curs = experiment_setting.Current;


[locs_numeric, uniq_locs] = grp2idx(locs(1:24));
[durs_numeric, uniq_durs] = grp2idx(durs(1:24));
[freqs_numeric, uniq_freqs] = grp2idx(freqs(1:24));
[curs_numeric, uniq_curs] = grp2idx(curs(1:24));

locs_numeric(isnan(locs_numeric))=0;
durs_numeric(isnan(durs_numeric))=0;
freqs_numeric(isnan(freqs_numeric))=0;
curs_numeric(isnan(curs_numeric))=0;
%% varying region of stim
[groups, id_durs, id_freqs, id_curs] = findgroups(durs_numeric, freqs_numeric, curs_numeric);
grp_locs = find_groups_appearing_twice(groups);

num_grp = size(grp_locs, 1);
trial_nums = ids(grp_locs(:,2:end));
durations = uniq_durs(id_durs(grp_locs(:,2)));
frequencies = uniq_freqs(id_freqs(grp_locs(:,2)));
currents = uniq_curs(id_curs(grp_locs(:,2)));

%%
A_numeric = locs_numeric;
A_uniq = uniq_locs;
B_numeric = durs_numeric;
B_uniq = uniq_durs;
C_numeric = freqs_numeric;
C_uniq = uniq_freqs;
D_numeric = curs_numeric;
D_uniq = uniq_curs;

Comp1 = findComp(A_numeric, B_numeric, C_numeric, D_numeric, ids, A_uniq, B_uniq, C_uniq, D_uniq );

%% varying current
Comp2 = findComp(D_numeric, B_numeric, C_numeric, A_numeric, ids, D_uniq, B_uniq, C_uniq, A_uniq );
%% varying duration
Comp3 = findComp(B_numeric, D_numeric, C_numeric, A_numeric, ids, B_uniq, D_uniq, C_uniq, A_uniq );
%% varying frequency
Comp4 = findComp(C_numeric, B_numeric, D_numeric, A_numeric, ids, C_uniq, B_uniq, D_uniq, A_uniq );
%%
function allComps = findComp(A_numeric, B_numeric, C_numeric, D_numeric, ids, A_uniq, B_uniq, C_uniq, D_uniq )
groups = findgroups(B_numeric, C_numeric, D_numeric);
grp_idx = find_groups_appearing_twice(groups);

allComps.trial_nums = ids(grp_idx(:,2:end));
allComps.B = B_uniq(B_numeric(grp_idx(:,2)));
allComps.C = C_uniq(C_numeric(grp_idx(:,2)));
allComps.D = D_uniq(D_numeric(grp_idx(:,2)));
allComps.A = A_uniq(A_numeric(grp_idx(:,2:end)));

end
%%
function result = find_groups_appearing_twice(lst)
    % Count occurrences of each number
    unique_vals = unique(lst);
    counts = histcounts(lst,'BinMethod','integers');
    % Find numbers that appear exactly twice
    numbers_twice = unique_vals(counts == 2);
    
    % Initialize result matrix
    result = zeros(length(numbers_twice), 3);
    
    % Find indices and store in result
    for i = 1:length(numbers_twice)
        num = numbers_twice(i);
        indices = find(lst == num);
        result(i, :) = [num, indices(1), indices(2)];
    end
end