function [time,est,input,output,parameters_out] = dynmout(fitsignals,time,est,parameters_out,bias)

%Because of the modification I did to fitsegi, this will set the 
%value of 'est' = to the model fit as plotted by dynamic
%analysis. However, this is the reconstructed version of the fit,
%and it needs to be modified in order to be compared with the real fr
%So, this function will reconstruct a proper signal, and it will
%then compute a VAF coefficient.


global M Md ns buf11 buf12 Plotlist

n = size(Md,1);
Mi=Md;  
for i = 1:n
   Mi(i,1)=M(ns(i),1); Mi(i,2)=M(ns(i),2);	
end 


%Reconstruct a full length version of the plotted estimate
%This is more or less the inverse of what we did in SlideMN

n_in = size(fitsignals,2);
time = fitsignals(:,1)';
for i = 2:(n_in - 1),
   input(i,:) = fitsignals(:,i)';
end
output = fitsignals(:,n_in)';

if isempty(bias)
   bias = 0;
end

est(length(est):length(est)+50) = bias;


assignin('base','est',est);
assignin('base','time',time);
assignin('base','input',input);
assignin('base','output',output);
assignin('base','parameters_out',parameters_out);


all = 1:length(est);
inter = find(est == bias);
all(inter) = [];
t_fr = output(all);
t_est = est(all);

%keyboard

%Here, we compute the VAF

var_fr = var(t_fr);  
var_error = var(t_est-t_fr');  
VAF = 1 - (var_error/var_fr);

%Display the results
disp(' ');
disp('______________________________________');
disp(['Number of saccades tested: ', int2str(n)]);
n_param = (size(parameters_out,2)-2)/2';
disp(' ')
disp(['Output signal: ',Plotlist(buf12,:)]);
disp(' ')
disp(['Gain +/- std for ',Plotlist(buf11,:),': ',num2str(parameters_out(1)),' +/- ',num2str(parameters_out(1+n_param))]);

if n_param == 4,
   i = 2;
   disp(['Gain +/- std for 1st derivative of ',Plotlist(buf11,:),': ',num2str(parameters_out(i)),' +/- ',num2str(parameters_out(i+n_param))]);    %disp(['Gain +/- std for derivative #',int2str(i-1),' of ',Plotlist(buf11,:),': ',num2str(parameters_out(i)),' +/- ',num2str(parameters_out(i+n_param))])
   i = i+1;
   disp(['Gain +/- std for slide term: ',num2str(parameters_out(i)),' +/- ',num2str(parameters_out(i+n_param))])
   i = i+1;
   disp(['Gain +/- std for bias: ',num2str(parameters_out(i)),' +/- ',num2str(parameters_out(i+n_param))])
elseif n_param == 3,
   i = 2;
   disp(['Gain +/- std for 1st derivative of ',Plotlist(buf11,:),': ',num2str(parameters_out(i)),' +/- ',num2str(parameters_out(i+n_param))]);    %disp(['Gain +/- std for derivative #',int2str(i-1),' of ',Plotlist(buf11,:),': ',num2str(parameters_out(i)),' +/- ',num2str(parameters_out(i+n_param))])
   i = i+1;
   disp(['Gain +/- std for bias: ',num2str(parameters_out(i)),' +/- ',num2str(parameters_out(i+n_param))])
elseif (n_param == 2) & (bias == 0),
   i = 2;
   disp(['Gain +/- std for 1st derivative of ',Plotlist(buf11,:),': ',num2str(parameters_out(i)),' +/- ',num2str(parameters_out(i+n_param))]);    %disp(['Gain +/- std for derivative #',int2str(i-1),' of ',Plotlist(buf11,:),': ',num2str(parameters_out(i)),' +/- ',num2str(parameters_out(i+n_param))])
elseif (n_param == 2) & (bias ~= 0)
   i = 2;
   disp(['Gain +/- std for bias: ',num2str(parameters_out(i)),' +/- ',num2str(parameters_out(i+n_param))])
end
disp(' ')
disp(['Mean error = ',num2str(parameters_out(size(parameters_out,2)-1))])
disp(['VAF =  ', num2str(VAF)])
disp(['BIC = ',num2str(parameters_out(size(parameters_out,2))) ])
disp(' ')









