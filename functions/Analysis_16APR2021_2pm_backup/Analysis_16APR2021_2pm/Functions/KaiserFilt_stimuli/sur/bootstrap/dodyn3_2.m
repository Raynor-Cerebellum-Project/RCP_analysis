function [x,se,error_sim,VAF,BIC,est,Others] = dodyn3_2(M,ns,lat,out1,in1to4,mode_flag)


%Must be used through dynatwo.m
%
%Mode_FLag: 0 or [] = simple estimate
%                 1 = prediction
%                 2 = bootstrap
%
%Output formatted in plotdyn.m
%
%NOTE: "Others" is a cool structure in which any additional information can be stored
%
%PAS, 2000

warning off 

tic;

x0 = [100 0 0 0 0];  %default initial conditions


%if (nargin == 5),               %default estimate if flag not specified
%    pred_flag = 0;
    %elseif (size(mode_flag,2) > 1) & (mode_flag(1) == 8888), %for estimation in bootstrap or latency estimation, where mode_flag == [8888 x0_modified]
%    x0 = mode_flag(2:6);
%    pred_flag = 0;
%elseif (size(mode_flag,2) > 1) & (mode_flag(1) ~= 8888), %for predictions, where mode_flag == x_pred
%    x = mode_flag;
%    pred_flag = 0;
%elseif (size(mode_flag,2) > 1) & (mode_flag(2) == 5678)  %for multi-units
    mode_flag = mode_flag;
    pred_flag = 0;
    %elseif (mode_flag == 9999),     %if the initial conditions need to be changed, with single estimation
   % x0 = getini(x0);   
   % pred_flag = 0;
   %elseif (mode_flag == 0) | (mode_flag == 1234),        %if flag already set to estimate
  %  pred_flag = 0;
  %else                            %default estimate if anything else
 %   pred_flag = 0;
 %end


%prepare the signals for the analysis process, i.e. only keep the valuable data

cut = [];
for i = 1:size(ns,2),
    cut = [cut M(ns(i),1):M(ns(i),2)];
end


t_in = [in1to4(cut+lat,1) in1to4(cut+lat,2) in1to4(cut+lat,3) in1to4(cut+lat,4)];
%if (mode_flag(1) == 1234), %i.e. output signal is not FR
    t_out = out1(cut);
    %t_out = out1(cut+lat);
    %else
    %t_out = out1(cut);
    %end
 

%define and initialize the "Others" output structure
Others = struct('n_imp',[],'x0',[],'NN',[],'time_array',[]);

Others.n_imp = [(sum(in1to4(:,1)) ~= 0) (sum(in1to4(:,2)) ~= 0) (sum(in1to4(:,3)) ~= 0) (sum(in1to4(:,4)) ~= 0)];
Others.time_array = cut; 

%Evaluate/Predict the dynamic models%
if (pred_flag ~= 1) & (out1(1) ~= 9999),
    options = optimset('display','off','MaxFunEvals',[25000],'MaxIter',[15000]);  %,'DiffMinChange',[1e-6]'TolFun',[1e-15],
    [x,RESNORM,RESIDUAL,EXITFLAG] = lsqnonlin('errmdyn2',x0,[],[],options,t_out,t_in);  
    Others.exit = EXITFLAG;

    x = [([ones(length(t_in),1) t_in] \ t_out)'];  %Jay's code, see bottom
    est = x(1) + x(2).*t_in(:,1) + x(3).*t_in(:,2) + x(4).*t_in(:,3) + x(5).*t_in(:,4 ); 
    se = sd_err(t_out,t_in,est,Others); %sub-function
    Others.exit = 111111;
    Others.x0 = x0;
elseif (pred_flag ~= 1),
    options = optimset('display','off');
    [x,RESNORM,RESIDUAL,EXITFLAG] = lsqnonlin('errmdyn2nb',x0,[],[],options,t_out,t_in);  
    Others.exit = EXITFLAG;

    x0(1) = [0];
    x = [([t_in] \ t_out)'];  %Jay's code, see bottom
    x = [0 x];
    est = x(1) + x(2).*t_in(:,1) + x(3).*t_in(:,2) + x(4).*t_in(:,3) + x(5).*t_in(:,4 ); 
    se = sd_err(t_out,t_in,est,Others); %sub-function
    Others.exit = 111111;
    Others.x0 = x0;
else
    est = x(1) + x(2).*t_in(:,1) + x(3).*t_in(:,2) + x(4).*t_in(:,3) + x(5).*t_in(:,4 );   
    se = [0 0 0 0 0];
    Others.exit = 9999;
    Others.x0 = [9999 9999 9999 9999 9999];
end

%error_sim = mean(t_out-est);
NN=length(t_out);
BIC = log(1/NN*(sum((t_out - est).^2))) + sum(Others.n_imp)/2*(log(NN))/NN;
VAF = 1 - (var(t_out - est)/var(t_out));

Others.timeit = toc;
Others.NN = NN;



%%%%%%%%%%%%%%%%%%%%%
%  Subfunction #1
%%%%%%%%%%%%%%%%%%%%%

function [SE_out] = sd_err(Y,X,Yhat,Others)

n = max(size(Y));
k = sum(Others.n_imp);

y = Y - mean(Y);
yhat = Yhat - mean(Y);
R2 = sum(yhat.^2) ./ sum(y.^2);
sy = sqrt( ( (1-R2).*sum(y.^2) ) ./ (n - k - 1) );

X1 = X;
z = find(Others.n_imp == 0);
X1(:,z) = [];
X1 = [X1 ones(size(X1(:,1)))];
c = inv(X1'*X1);

V = []; SE = [];

%Get bias first
for i = 1:k+1,
    V(i)  = (c(i,i)).*sy.^2;
    SE(i) = sqrt(c(i,i)).*sy;
end

%Get the other parameters
z = find(Others.n_imp == 1);
SE_out = [SE(i) 0 0 0 0];
for j = 1:sum(Others.n_imp)
    SE_out(z(j)+1) = SE(j);
end
