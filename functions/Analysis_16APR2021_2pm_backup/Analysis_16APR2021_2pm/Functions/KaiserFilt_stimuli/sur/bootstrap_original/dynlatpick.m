% This m-file fits the dynamic latency
% using model 2d and conjugated position
%
%PAS, 2000

%Make latency array

disp('Chut!! I am thinking...')
disp(' ')
startpt = -15;		%set range of latencies to check for
endpt = 35;
sizelistlat = abs(startpt) + endpt + 1;
sta = startpt;

results = zeros(sizelistlat,2);

for r = 1:sizelistlat
    results(r,1) = sta;
    sta = sta + 1;
end;

if (mode_flag == 9999),
    mode_flag = [8888 getini([100 0 0 0 0])]; 
end

%Set the latency and replot

warning off
fig1 = findobj('name','Latency Results');
if (isempty(fig1)),
    lat_fig = figure('name','Latency Results','number','off','position',[232   103   515   619]);
    axis([startpt,endpt,0,1]);
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

for w = 1:sizelistlat
    M = M + lat - results(w,1);
    lat = results(w,1); 
    
    [x,se,error_sim,VAF,BIC,est,Others] = dodyn3(M,ns,lat,out1,in1to4,mode_flag);       
    results(w,2) = VAF;		%write the VAF in the second column with corresponding latency latency
    plot(lat,results(w,2),'oy');
    pause(0.1)
end;
warning on
hold off

bestlatindice = find(results(:,2) == max(results(:,2)));
bestlat4 = results(bestlatindice,1);

std_VAF = std(results(:,2));

%display and plot the results
disp(['The optimal dynamic lead time is: ',num2str(bestlat4),' msec'])
fig2 = findobj('name','Latency Results');
figure(fig2); 
title(['The optimal dynamic lead time is: ',num2str(bestlat4),' msec;'])
hold on
plot(results(bestlatindice,1),results(bestlatindice,2),'*r')
hold off

M = M + lat - bestlat4;
lat = bestlat4;
replot;

[x,sd,error_sim,VAF,BIC,est,Others] = dodyn3(M,ns,lat,out1,in1to4,mode_flag);
plotdyn;


clear startpt endpt sizelistlat r w sta bestlat bestlatindice
