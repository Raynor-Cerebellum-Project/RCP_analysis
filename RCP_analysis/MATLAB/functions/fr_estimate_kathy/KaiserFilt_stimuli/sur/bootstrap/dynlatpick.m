function [x,sd,error_sim,VAF,BIC,est,Others] = dynlatpick(M,ns,lat,out1,in1to4,mode_flag);
% This m-file fits the dynamic latency
% using model 2d and conjugated position
%
%PAS, 2000


%Make latency array

sampling_frequency = [0.6, 1, 2, 4, 8, 16];
sam_freq = sampling_frequency(6); 
%Set parameters

startpt = evalin('base', 'lat_min'); %set range of latencies to check for
%endpt = 20;
endpt = evalin('base', 'lat_max');


if M(1,1)<endpt
    error('Start of the first M was less than the maximum latency, please select another M.')    
end
%startpt =-min(floor(250/sam_freq),300);		%set range of latencies to check for
% endpt =0;
% startpt =-100;		%set range of latencies to check for CHANGED FOR EYE ANALYSIS - AD 10/2011
%endpt =min(floor(750/sam_freq),300);
sizelistlat = endpt - startpt + 1;
sta = startpt;

results = zeros(sizelistlat,2);
for r = 1:sizelistlat
    results(r,1) = sta;
    sta = sta + 1;
end;

if (mode_flag == 9999),
    mode_flag = [8888 getini([100 0 0 0 0])]; 
end


%Plot progress
if  (size(mode_flag,2) == 1) | ((size(mode_flag,2) > 1) & (mode_flag(2) ~= 5678)),
    warning off
    fig1 = findobj('name','Latency Results');
    if (isempty(fig1)),
        lat_fig = figure('name','Latency Results','position',[232   53   515   619]);
        %axis([startpt,endpt,0,1]);
        grid on
        xlabel('FR Lead Time (msec)');
        ylabel('VAF');
        title(['Dynamic Latency Plot']);
    else
        figure(fig1);
        clf;   
    end;
    hold on
    pause(1)  
end



%%%%%%%%%%
% Main Latency LOOP

for w = 1:sizelistlat
    M = M + lat - results(w,1);
    lat = results(w,1); 

    [x,se,error_sim,VAF,BIC,est,Others] = dodyn3(M,ns,lat,out1,in1to4,mode_flag);       
    results(w,2) = VAF;		%write the VAF in the second column with corresponding latency latency
    if  (size(mode_flag,2) == 1) | ((size(mode_flag,2) > 1) & (mode_flag(2) ~= 5678)),
        plot(lat,results(w,2),'b*');
        pause(0.1)
    end
end;
warning on
hold off

bestlatindice = find(results(:,2) == max(results(:,2)));
bestlat4 = results(bestlatindice,1);

std_VAF = std(results(:,2));

%%%%%%%%%%%%

if  (size(mode_flag,2) == 1) | ((size(mode_flag,2) > 1) & (mode_flag(2) ~= 5678)),
    
    %display and plot the results
    disp(' -----------------------------------------------------------------------------------------------------')
    disp(['The optimal dynamic lead time is: ',num2str(bestlat4),' msec'])
    fig2 = findobj('name','Latency Results');
    figure(fig2); 
    title(['The optimal dynamic lead time is: ',num2str(bestlat4),' msec;'])
    hold on
    plot(results(bestlatindice,1),results(bestlatindice,2),'pr')
    hold off
    
end




M = M + lat - bestlat4;
lat = bestlat4;

[x,sd,error_sim,VAF,BIC,est,Others] = dodyn3(M,ns,lat,out1,in1to4,mode_flag);
Others.lat = bestlat4;
save('latest_fit.mat','x','sd','error_sim','VAF','BIC','est','lat')


%evalin('base','plotdyn;')



