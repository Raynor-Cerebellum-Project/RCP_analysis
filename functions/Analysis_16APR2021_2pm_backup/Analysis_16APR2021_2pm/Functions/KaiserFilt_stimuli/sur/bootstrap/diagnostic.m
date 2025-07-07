function [DIAG] = diagnostic(M,ns,lat,out1,in1to4,model,mode_flag)

%Performs a Jack-Knife analysis of the currently displayed segments.
%Plots the parameters obtained when each segment number is removed.
%Use to remove segments that bias the overall estimate

tic

disp(' ')
disp('Please be patient...')
disp(' ')

DIAG = [];
ns_ini = ns;

x0 = [100 0 0 0 0];  %default initial conditions

if (mode_flag == 9999),         %in the case where you run a single bootstrap run
    mode_flag = [8888 getini([100 0 0 0 0])]; 
elseif (size(mode_flag,2) > 1) & (mode_flag(1) == 8888)  %for multi-units
    mode_flag = mode_flag;
else
    mode_flag = 0;
end


%First, get values on full data set
[x,sd,error_sim,VAF,BIC,est,Others] = dodyn3(M,ns_ini,lat,out1,in1to4,mode_flag);
DIAG(1,:) = [x sd VAF BIC error_sim];

%Second, compute the Jack-Knife estimates
for i = 1:max(size(ns_ini)),
    ns_new = ns_ini;
    ns_new(i) = [];
    [x,sd,error_sim,VAF,BIC,est,Others] = dodyn3(M,ns_new,lat,out1,in1to4,mode_flag);
    DIAG(i+1,:) = [x sd VAF BIC error_sim];
end

%Finally, plot the results
diag_p = figure('position',[21    82   988   637],'name','Bootstrap Diagnosis','PaperOrientation', 'landscape', 'PaperPosition' , [0.25 0.25 10.5 8]);

%Bias
n=1;
subplot(3,3,1)
plot(ns_ini,DIAG(2:size(DIAG,1),n) - DIAG(1,n),'*r')
title('BIAS')
ylabel('B(i removed) - B(all)')
xrange = get(gca,'xlim');
p_top = line(xrange,[2.5*std(DIAG(2:size(DIAG,1),n)) 2.5*std(DIAG(2:size(DIAG,1),n))]);
p_bottom = line(xrange,[-2.5*std(DIAG(2:size(DIAG,1),n)) -2.5*std(DIAG(2:size(DIAG,1),n))]);
set(p_top,'color',[0 1 0],'LineStyle',':')
set(p_bottom,'color',[0 1 0],'LineStyle',':')

%1st input
n=2;
subplot(3,3,4) 
plot(ns_ini,DIAG(2:size(DIAG,1),n) - DIAG(1,n),'*r')
title(['1st input: ' model(n-1,:)])   
ylabel('B(i removed) - B(all)')
xrange = get(gca,'xlim');
p_top = line(xrange,[2.5*std(DIAG(2:size(DIAG,1),n)) 2.5*std(DIAG(2:size(DIAG,1),n))]);
p_bottom = line(xrange,[-2.5*std(DIAG(2:size(DIAG,1),n)) -2.5*std(DIAG(2:size(DIAG,1),n))]);
set(p_top,'color',[0 1 0],'LineStyle',':')
set(p_bottom,'color',[0 1 0],'LineStyle',':')

%1st input option
n=3;
subplot(3,3,7) 
plot(ns_ini,DIAG(2:size(DIAG,1),n) - DIAG(1,n),'*r')
title(['1st input option: ' model(n-1,:)])   
ylabel('B(i removed) - B(all)')
xlabel('ns(i)')
xrange = get(gca,'xlim');
p_top = line(xrange,[2.5*std(DIAG(2:size(DIAG,1),n)) 2.5*std(DIAG(2:size(DIAG,1),n))]);
p_bottom = line(xrange,[-2.5*std(DIAG(2:size(DIAG,1),n)) -2.5*std(DIAG(2:size(DIAG,1),n))]);
set(p_top,'color',[0 1 0],'LineStyle',':')
set(p_bottom,'color',[0 1 0],'LineStyle',':')

%2nd input
n=4;
subplot(3,3,5) 
plot(ns_ini,DIAG(2:size(DIAG,1),n) - DIAG(1,n),'*r')
title(['2nd input: ' model(n-1,:)])   
ylabel('B(i removed) - B(all)')
xrange = get(gca,'xlim');
p_top = line(xrange,[2.5*std(DIAG(2:size(DIAG,1),n)) 2.5*std(DIAG(2:size(DIAG,1),n))]);
p_bottom = line(xrange,[-2.5*std(DIAG(2:size(DIAG,1),n)) -2.5*std(DIAG(2:size(DIAG,1),n))]);
set(p_top,'color',[0 1 0],'LineStyle',':')
set(p_bottom,'color',[0 1 0],'LineStyle',':')

%2nd input option
n=5;
subplot(3,3,8) 
plot(ns_ini,DIAG(2:size(DIAG,1),n) - DIAG(1,n),'*r')
title(['2nd input option: ' model(n-1,:)])   
ylabel('B(i removed) - B(all)')
xlabel('ns(i)')
xrange = get(gca,'xlim');
p_top = line(xrange,[2.5*std(DIAG(2:size(DIAG,1),n)) 2.5*std(DIAG(2:size(DIAG,1),n))]);
p_bottom = line(xrange,[-2.5*std(DIAG(2:size(DIAG,1),n)) -2.5*std(DIAG(2:size(DIAG,1),n))]);
set(p_top,'color',[0 1 0],'LineStyle',':')
set(p_bottom,'color',[0 1 0],'LineStyle',':')


%VAF
n=11;
subplot(3,3,6) 
plot(ns_ini,DIAG(2:size(DIAG,1),n) - DIAG(1,n),'*r')
title('VAF')   
ylabel('B(i removed) - B(all)')
xrange = get(gca,'xlim');
p_top = line(xrange,[2.5*std(DIAG(2:size(DIAG,1),n)) 2.5*std(DIAG(2:size(DIAG,1),n))]);
p_bottom = line(xrange,[-2.5*std(DIAG(2:size(DIAG,1),n)) -2.5*std(DIAG(2:size(DIAG,1),n))]);
set(p_top,'color',[0 1 0],'LineStyle',':')
set(p_bottom,'color',[0 1 0],'LineStyle',':')

%BIC
n=12;
subplot(3,3,9) 
plot(ns_ini,DIAG(2:size(DIAG,1),n) - DIAG(1,n),'*r')
title('BIC')   
ylabel('B(i removed) - B(all)')
xlabel('ns(i)')
xrange = get(gca,'xlim');
p_top = line(xrange,[2.5*std(DIAG(2:size(DIAG,1),n)) 2.5*std(DIAG(2:size(DIAG,1),n))]);
p_bottom = line(xrange,[-2.5*std(DIAG(2:size(DIAG,1),n)) -2.5*std(DIAG(2:size(DIAG,1),n))]);
set(p_top,'color',[0 1 0],'LineStyle',':')
set(p_bottom,'color',[0 1 0],'LineStyle',':')

streamer('Diagnostic Jack-Knife Analysis')


x= toc;

disp(' ')
disp('Dotted lines denote +/- 2.5 STD')
disp(['Elapsed time (sec): ',num2str(x)])