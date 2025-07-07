function [x,sd,error_sim,VAF,BIC,est,Others] = dynlatpick2(M,ns,lat,out1,in1to4,mode_flag);
% This m-file fits the dynamic latency
% using model 2d and conjugated position
%
%PAS, 2000

%modified SSG, 2003
%==================================================================================

format short g;

mode_flag=1234;
%latency_all=[];
%gain_h=[]; bestgain=[]; gain_h_all=[];
values=[];

for i=1:length(ns);
    startpt = -7;		%set range of latencies to check for
    endpt = 5;
    sizelistlat = endpt - startpt + 1;
    sta = startpt;
    
    results = zeros(sizelistlat,5);
    for r = 1:sizelistlat
        results(r,2) = sta;
        sta = sta + 1;
    end;
    
    for w = 1:sizelistlat
        M = M + lat - results(w,2);
        lat = results(w,2); 

        [x,se,error_sim,VAF,BIC,est,Others] = dodyn3(M,ns(i),lat,out1,in1to4,mode_flag);     
        %        gain_h=[gain_h; x];
        results(w,1)=ns(i);
        results(w,5) = VAF;
        results(w,4) = x(1);
        results(w,3) = x(2);
    end;
    warning on

%    keyboard
    bestlatindice = find(results(:,5) == max(results(:,5)));

    values=[values; results(bestlatindice,:)];

    %    bestgain=gain_h(bestlatindice,:);
    %    gain_h_all=[gain_h_all; bestgain];
%    keyboard
end    
sz2=size(values);
sz3=sz2(1);

values=values;

mean_latency=mean(values(:,2));
%SD_latency=std(values(:,2));
SE_latency=std(values(:,2))/(sqrt(sz3));

mean_hhv_gain= - mean(values(:,3));
%SD_hhv_gain=std(values(:,3));
SE_hhv_gain=std(values(:,3))/(sqrt(sz3));

mean_bias=mean(values(:,4));
SE_bias=std(values(:,4))/(sqrt(sz3));

%latency_all
%mean_latency=mean(latency_all)
%SD_latency=std(latency_all)
%hhv_gain_all=gain_h_all(:,4)
%mean_hhv_gain=-mean(gain_h_all(:,4))
%SD_hhv_gain=std(gain_h_all(:,4))
%mean_phase=(360*f*mean_latency)/1000
%SD_phase=(360*f*SD_latency)/1000

%%%%%%%%%%%%

disp(sprintf('model:\t\t ehv = bias + hhv'))
disp('--------------------------------------------------------------');
disp(sprintf('ns\t latency\t hhv_gain\t Bias\t VAF'))


for i = 1:sz3,
        disp(sprintf('%0.4f\t %0.4f\t %0.4f\t %0.4f\t %0.4f', values(i,:))); 
end

disp('--------------------------------------------------------------');
disp(sprintf('mean\t %0.4f\t %0.4f\t %0.4f', mean_latency, mean_hhv_gain, mean_bias)); 
disp(sprintf('SE\t %0.4f\t %0.4f\t %0.4f', SE_latency, SE_hhv_gain, SE_bias)); 
