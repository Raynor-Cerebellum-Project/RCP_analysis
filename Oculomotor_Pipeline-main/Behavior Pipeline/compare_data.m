%% compare error between two results (freq filter first and freq filter last)
% P1 is the result from pipeline 1 (freq filter first) and P2 is the result
% from pipeline 2 (freq filter last)
function error = compare_data(D0, P1, P2)
    
    P1_kept = P1{2};
    P2_kept = P2{2};
    n = size(D0.ehp_left,1);
    n1 = length(P1_kept.list)+1*(isempty(P1_kept.list));
    n1 =n1/n;
    n2 = length(P2_kept.list)+1*(isempty(P2_kept.list));
    n2 = n2/n;
    if (~isempty(P1_kept.list) & ~isempty(P2_kept.list))
        r1 = tot_SNR(D0, P1_kept);
        r2 = tot_SNR(D0, P2_kept);
    elseif ~isempty(P2_kept.list)
        r1 = tot_SNR(D0, []);
        r2 = tot_SNR(D0, P2_kept);
    else 
        r1 = tot_SNR(D0, P1_kept);
        r2 = tot_SNR(D0, []);
    end
    error = [n1*(r1.all), n2*(r2.all)];
end

function r = tot_SNR(d0, d1)
    if ~isempty(d1)
    n = length(d1.list);
    r_H_left = zeros(n,2);
    r_H_right = zeros(n,2);
    r_V_left = zeros(n,2);
    r_V_right = zeros(n,2);

    for j = 1:n
        i = d1.list(j);
        r_H_left(j, 1)= snr(d0.ehp_left{i},d0.ehp_left{i}-d1.ehp_left{j});
        r_H_right(j, 1) = snr(d0.ehp_right{i},d0.ehp_right{i}-d1.ehp_right{j});
        r_V_left(j, 1) = snr(d0.evp_left{i},d0.evp_left{i}-d1.evp_left{j});
        r_V_right(j, 1) = snr(d0.evp_right{i},d0.evp_right{i}-d1.evp_right{j});
        r_H_left(j, 2)= snr(d0.ehv_left{i},d0.ehv_left{i}-d1.ehv_left{j});
        r_H_right(j, 2) = snr(d0.ehv_right{i},d0.ehv_right{i}-d1.ehv_right{j});
        r_V_left(j, 2) = snr(d0.evv_left{i},d0.evv_left{i}-d1.evv_left{j});
        r_V_right(j, 2) = snr(d0.evv_right{i},d0.evv_right{i}-d1.evv_right{j});
    end
    r.r_H_left = min(median(r_H_left));
    r.r_H_right = min(median(r_H_right));
    r.r_V_left = min(median(r_V_left));
    r.r_V_right = min(median(r_V_right));
    
    r.all = median([r.r_H_left, r.r_H_right, r.r_V_left, r.r_V_right], 'all');
    
    else
        r.r_H_left = 0;
        r.r_H_right = 0;
        r.r_V_left = 0;
        r.r_V_right = 0;
        r.all = 0;
        
    end
end