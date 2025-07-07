function [CI_BCa] = BS_conf(ORIG,DATA,M,ns,lat,out1,in1to4,userdir,alpha)

%As per Carpenter and Bithell 2000
%PAS, 2000


%First, get the Jack-Knife estimate for a_BCa
a_BCa = []; JK = [];
DATA(size(DATA,1)-1:size(DATA,1),:) = [];
CI_BCa = zeros(2,size(DATA,2));

ns_ini = ns;
for i = 1:max(size(ns_ini)),
    ns = ns_ini;
    ns(i) = [];
    [x,sd,error_sim,VAF,BIC,est,n_imp] = dodyn2(M,ns,lat,out1,in1to4,2);
    thi(i,:) = [x sd VAF BIC error_sim];
end
tt = mean(thi);

for i = 1:size(tt,2),
    if (sum(DATA(:,i))) ~= size(DATA,1) | (sum(DATA(:,i)) ~= 0),
        a_BCa(i) = ( sum( (tt(i) - thi(:,i)).^3 ) )/ ( 6* (sum( (tt(i) - thi(:,i)).^2)).^1.5 );
    end
end


%NOW, we can get the confidence intervals
for i = 1:5,
    if (sum(DATA(:,i))) == size(DATA,1) | (sum(DATA(:,i)) == 0),
        CI_BCa(:,i) = [0 ; 0];
    else
        th  = ORIG(i);                   %parameter estimate from original dataset (theta hat)
        ths = DATA(:,i);                    %sample of parameter estimates from Bootstrap runs (theta hat star)
        B = max(size(ths));     %number of bootstrap iterations
        ths_o = sort(ths);         %ts sorted in incremental order
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %The Bias Corrected and Accelerated Method (p.1153 of Carpenter and Bithell 2000)
        
        p = size(find(ths_o < th),1);
        b_BCa = norminv(p/B);
        
        z = norminv(alpha/2);
        Q = floor( (B + 1) * normcdf(b_BCa - ( (z - b_BCa) / (1 + a_BCa(i)*(z - b_BCa)) )) );
        
        a1 = floor(Q); a2 = ceil(Q);
        if (a1 ~= a2),
            UB_BCa = ths_o(a1) + ( (norminv(Q/(B+1)) - norminv(a1/(B+1))) / (norminv(a2/(B+1)) - norminv(a1/(B+1))) ) * (ths_o(a2)-ths_o(a1)); %Thsq
            LB_BCa = ths_o(a1) - ( (norminv(Q/(B+1)) - norminv(a1/(B+1))) / (norminv(a2/(B+1)) - norminv(a1/(B+1))) ) * (ths_o(a2)-ths_o(a1)); %Thsq
        else
            UB_BCa = ths_o(Q); %Thsq
            LB_BCa = th - (ths_o(Q) - th); %Thsq
        end
        
        CI_BCa(:,i) = [LB_BCa ; UB_BCa];
    end
end

keyboard

for i = 11:13,
    if (sum(DATA(:,i))) == size(DATA,1) | (sum(DATA(:,i)) == 0),
        CI_BCa(:,i) = [0 ; 0]; 
    else
        th  = ORIG(i);                   %parameter estimate from original dataset (theta hat)
        ths = DATA(:,i);                    %sample of parameter estimates from Bootstrap runs (theta hat star)
        B = max(size(ths));     %number of bootstrap iterations
        ths_o = sort(ths);         %ts sorted in incremental order
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %The Bias Corrected and Accelerated Method (p.1153 of Carpenter and Bithell 2000)
        
        p = size(find(ths_o < th),1);
        b_BCa = norminv(p/B);
        
        z = norminv(alpha/2);
        Q = floor( (B + 1) * normcdf(b_BCa - ( (z - b_BCa) / (1 + a_BCa(i)*(z - b_BCa)) )) );
        
        a1 = floor(Q); a2 = ceil(Q);
        if (a1 ~= a2),
            UB_BCa = ths_o(a1) + ( (norminv(Q/(B+1)) - norminv(a1/(B+1))) / (norminv(a2/(B+1)) - norminv(a1/(B+1))) ) * (ths_o(a2)-ths_o(a1)); %Thsq
            LB_BCa = ths_o(a1) - ( (norminv(Q/(B+1)) - norminv(a1/(B+1))) / (norminv(a2/(B+1)) - norminv(a1/(B+1))) ) * (ths_o(a2)-ths_o(a1)); %Thsq
        else
            UB_BCa = ths_o(Q); %Thsq
            LB_BCa = th - (ths_o(Q) - th); %Thsq
        end
        
        CI_BCa(:,i) = [LB_BCa ; UB_BCa];   
    end
end









if 0,  %WILL NOT BE EXECUTED (STORAGE)
    
    th  = ORIG(1);                   %parameter estimate from original dataset (theta hat)
    ths = DATA(:,1);                    %sample of parameter estimates from Bootstrap runs (theta hat star)
    B = max(size(ths));     %number of bootstrap iterations
    ths_o = sort(ths);         %ts sorted in incremental order
    alpha = 0.05;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %The Percentile Method (p.1152 of Carpenter and Bithell 2000)
    
    ind = floor((1 - alpha) * (B + 1));
    UB_perc = ths_o(ind);
    
    disp(' ')
    disp('_________________________________________________________')
    disp('Percentile Method (p.1152 of Carpenter and Bithell 2000)')
    disp('Alpha:')
    disp(alpha)
    disp('Rough Estimate of (-inf, UB):')
    disp(UB_perc)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %The Bias Corrected Method (p.1153 of Carpenter and Bithell 2000)
    
    p = size(find(ths_o < th),1);
    b_BC = norminv(p/B);
    
    z = norminv(alpha);
    Q = (B + 1) * normcdf(2*b_BC - z);
    
    UB_BC_rough = ths_o(floor(Q));
    
    a1 = floor(Q); a2 = ceil(Q);
    UB_BC_finer = ths_o(a1) + ( (norminv(Q/(B+1)) - norminv(a1/(B+1))) / (norminv(a2/(B+1)) - norminv(a1/(B+1))) ) * (ths_o(a2)-ths_o(a1)); %Thsq
    
    disp(' ')
    disp('_____________________________________________________________')
    disp('Bias Corrected Method (p.1153 of Carpenter and Bithell 2000)')
    disp('Alpha:')
    disp(alpha)
    disp('Rough Estimate of (-inf, UB):')
    disp(UB_BC_rough)
    disp('Finer Estimate of (-inf, UB):')
    disp(UB_BC_finer)
    
end

