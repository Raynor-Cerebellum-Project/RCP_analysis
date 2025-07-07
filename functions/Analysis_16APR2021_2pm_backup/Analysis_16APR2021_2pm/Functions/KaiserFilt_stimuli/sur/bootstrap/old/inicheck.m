% This m-file fits the dynamic latency
% using model 2d and conjugated position
%
%PAS, 2000

%Make latency array

disp('Chut!! I am thinking...')
disp(' ')
j=1; ini = [];
for i = -3.5:0.5:3.5,
    ini(j,:) = [100 i i i i 0];
    j = j+1;
end

%Set the latency and replot

warning off
fig1 = findobj('name','Initial Conditions Results');
if (isempty(fig1)),
    lat_fig = figure('name','Initial Conditions Results','position',[232   103   515   619]);
    axis([ini(1,2),ini(size(ini,1),2),0,1]);
    grid on
    xlabel('Parameters #2-5 value (Bias = 100)');
    ylabel('VAF');
    title(['Initial Conditions Plot']);
else
    figure(fig1);
    clf;   
end;
hold on
pause(1)

outini = [];
for w = 1:size(ini,1),
    mode_flag = [8888 ini(w,1:5)];
    
    [x,se,error_sim,VAF,BIC,est,Others] = dodyn3(M,ns,lat,out1,in1to4,mode_flag);       
    outini(w,:) = [ini(w,1:5) VAF x];		%write the VAF in the second column with corresponding latency latency
    plot(ini(w,2),VAF,'oy');
    pause(0.1)
end;
warning on
hold off

disp(' ')
disp(' ')
disp('Here are the results:')
disp(' ')
sprintf('Bias\tP(1)\tP(2)\tP(3)\tP(4)\tVAF')
disp(outini)

clear i j lat_fig mode_flag x se error_sim VAF BIC est Others ini

