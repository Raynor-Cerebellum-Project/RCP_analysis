

function [x,sd,error_sim,VAF,BIC,est,Others] = test(M,ns,lat,out1,in1to4,mode_flag);

latency_all=[];
for ns=1:length(ns);
    [x,sd,error_sim,VAF,BIC,est,Others] = dynlatpick(M,ns,lat,out1,in1to4,mode_flag);
    latency_all=[latency_all; Others.lat];
end

mean_latency=mean(latency_all)
SD_latency=std(latency_all)
