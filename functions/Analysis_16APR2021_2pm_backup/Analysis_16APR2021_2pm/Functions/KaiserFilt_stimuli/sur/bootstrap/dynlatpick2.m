function [x,sd,error_sim,VAF,BIC,est,Others] = dynlatpick2(M,ns,lat,out1,in1to4,mode_flag);
% This m-file fits the dynamic latency
% using model 2d and conjugated position
%
%PAS, 2000

%Make latency array



%Set parameters
startpt = -50;		%set range of latencies to check for
endpt = 15;
sizelistlat = endpt - startpt + 1;
sta = startpt;

results = zeros(sizelistlat,2);
for r = 1:sizelistlat
    results(r,1) = sta;
    sta = sta + 1;
end;

%if (mode_flag == 9999),
%    mode_flag = [8888 getini([100 0 0 0 0])]; 
%end


%Plot progress
%if  (size(mode_flag,2) == 1) | ((size(mode_flag,2) > 1) & (mode_flag(2) ~= 5678)),
%    warning off
%    fig1 = findobj('name','Latency Results');
%    if (isempty(fig1)),
%        lat_fig = figure('name','Latency Results','number','off','position',[232   103   515   619]);
%        axis([startpt,endpt,0,1]);
%        grid on
%        xlabel('FR Lead Time (msec)');
%        ylabel('VAF');
%        title(['Dynamic Latency Plot']);
%    else
%        figure(fig1);
%        clf;   
%    end;
%    hold on
%    pause(1)  
%end




%%%%%%%%%%
% Main Latency LOOP
mode_flag=1234;
latency_all=[];
for ns=1:length(M);
    for w = 1:sizelistlat
        M = M + lat - results(w,1);
        lat = results(w,1); 
        
        [x,se,error_sim,VAF,BIC,est,Others] = dodyn3(M,ns,lat,out1,in1to4,mode_flag);       
        results(w,2) = VAF;		%write the VAF in the second column with corresponding latency latency
        %    if  (size(mode_flag,2) == 1) | ((size(mode_flag,2) > 1) & (mode_flag(2) ~= 5678)),
        %        plot(lat,results(w,2),'oy');
        %        pause(0.1)
        %    end
    end;
    warning on
    %hold off
    
    bestlatindice = find(results(:,2) == max(results(:,2)));
    bestlat4 = results(bestlatindice,1);
    
    std_VAF = std(results(:,2));
    
    latency_all=[latency_all; bestlat4];
end    

latency_all
mean_latency=mean(latency_all)
SD_latency=std(latency_all)

%mean_phase=(360*f*mean_latency)/1000
%SD_phase=(360*f*SD_latency)/1000
%%%%%%%%%%%%




%if  (size(mode_flag,2) == 1) | ((size(mode_flag,2) > 1) & (mode_flag(2) ~= 5678)),
%    
%    %display and plot the results
%    disp(['The optimal dynamic lead time is: ',num2str(bestlat4),' msec'])
%    fig2 = findobj('name','Latency Results');
%    figure(fig2); 
%    title(['The optimal dynamic lead time is: ',num2str(bestlat4),' msec;'])
%    hold on
%    plot(results(bestlatindice,1),results(bestlatindice,2),'*r')
%    hold off
%    
%end

M = M + lat - bestlat4;
lat = bestlat4;

%[x,sd,error_sim,VAF,BIC,est,Others] = dodyn3(M,ns,lat,out1,in1to4,mode_flag);
%Others.lat = bestlat4;

%evalin('base','plotdyn;')



