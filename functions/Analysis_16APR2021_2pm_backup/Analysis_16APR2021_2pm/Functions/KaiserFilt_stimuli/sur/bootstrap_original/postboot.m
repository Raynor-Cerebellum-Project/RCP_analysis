function [] = postboot(filename_m,M,ns,lat,out1,in1to4,userdir,Others,print_flag)

%NOTE: calling this function without arguments prompts the user for
%      the file name where the already analyzed data is saved, and
%      replots / re-saves the data
%
%PAS, 2000




disp(' ')
disp('Post-Bootstrap Analysis...')
disp(' ')

model = ['---';'---';'---';'---';'---'];

if (nargin == 0),
    
    [fname,pname] = uigetfile('*.mat','Select Output *.MAT Bootstrap Data File');
    load([fname(4:length(fname)-6) '.mat'])
    filename_m = [fname(1:length(fname)-4) '.mat'];
    load(filename_m)
    print_flag = 0;
    
else
    
    load(filename_m)
    
end    

if ~exist('mode_flag')
    mode_flag = 0;
end

if ~exist('Others')
    Others.n_imp = [(sum(in1to4(:,1)) ~= 0) (sum(in1to4(:,2)) ~= 0) (sum(in1to4(:,3)) ~= 0) (sum(in1to4(:,4)) ~= 0)];
    Others.exit = 66666;
    Others.x0 = [66666 66666 66666 66666 66666];
    convperc = 99999;
end


for i = 1:4,
    if (Others.n_imp(i) == 0),
        DATA(:,i+1) = 0 .* DATA(:,i+1);
        ORIG(i+1) = 0;
    end
end



%GET THE BOOTSTRAP CONFIDENCE INTERVALS WITH THE BCa METHOD
alpha = 0.05;
z = -norminv(alpha/2);  %two sided

CI_BCa(:,:) = BS_conf(ORIG,DATA,M,ns,lat,out1,in1to4,mode_flag,userdir,alpha); %**sub-function


%PERFORM THE BOOTSTRAP DIAGNOSTIC TEST
[mean_check] = BS_diagnose(DATA,obs,ns);  %sub-function


%SAVE THE ANALYSIS RESULTS
filename_a = [filename_m(1:length(filename_m)-4) '_post.mat'];
save(filename_a,'CI_BCa','mean_check','alpha','z')

%SAVE A FORMATTED SUMMARY FILE

filename_sum = [filename_m(1:length(filename_m)-4) '_summary.txt'];

fid = fopen(filename_sum,'wt'); 
fprintf(fid,'__________________________________________________________________\n');
fprintf(fid,'MODEL:  \n\n');
fprintf(fid,'Filename: %s \n\n',filename_m(1:length(filename_m)-4) );
fprintf(fid,'%s = bias + %s + %s + %s + %s\n', model(5,:),model(1,:),model(2,:),model(3,:),model(4,:) );
fprintf(fid,'\n');
fprintf(fid,'alpha  =  %0.4f\n',alpha);
fprintf(fid,'N (bs) =  %0.4f\t\t\t N (sacc) =  %0.4f\n',size(DATA,1)-2,size(ns,2));
fprintf(fid,'Mean iteration time =  %0.4f\n',mean(time));
fprintf(fid,'\n');
fprintf(fid,'__________________________________________________________________\n');
fprintf(fid,'Original Data Set Estimates:  (parameter, s.e., 95%% LB conf., 95%% UB conf.)\n\n');
fprintf(fid,'\t bias\t %0.4f\t %0.4f\t %0.4f\t %0.4f\n', ORIG(1), ORIG(6) , ORIG(1)-z*ORIG(6) , ORIG(1)+z*ORIG(6));
for i = 2:5,
    fprintf(fid,'\t %s\t %0.4f\t %0.4f\t %0.4f\t %0.4f\n', model(i-1,:) , ORIG(i), ORIG(i+5) , ORIG(i)-z*ORIG(i+5) , ORIG(i)+z*ORIG(i+5));
end
fprintf(fid,'\n');
fprintf(fid,'\t VAF\t %0.4f\n', ORIG(11));
fprintf(fid,'\t BIC\t %0.4f\n', ORIG(12));
fprintf(fid,'\t Error\t %0.4f\n', ORIG(13));
fprintf(fid,'\n\n');

fprintf(fid,'__________________________________________________________________\n');
fprintf(fid,'Bootstrap Confidence Intervals:  (parameter, 95%% LB conf., 95%% UB conf.)\n\n');
fprintf(fid,'\t bias\t %0.4f\t \t %0.4f\t %0.4f\n', ORIG(1), CI_BCa(1,1) , CI_BCa(2,1));
for i = 2:5,
    fprintf(fid,'\t %s\t %0.4f\t \t %0.4f\t %0.4f\n', model(i-1,:), ORIG(i), CI_BCa(1,i) , CI_BCa(2,i));
end
fprintf(fid,'\n');
fprintf(fid,'\t VAF\t %0.4f\t \t %0.4f\t %0.4f\n', ORIG(11), CI_BCa(1,11) , CI_BCa(2,11));
fprintf(fid,'\t BIC\t %0.4f\t \t %0.4f\t %0.4f\n', ORIG(12), CI_BCa(1,12) , CI_BCa(2,12));
fprintf(fid,'\t Error\t %0.4f\t \t %0.4f\t %0.4f\n', ORIG(13), CI_BCa(1,13) , CI_BCa(2,13));
fprintf(fid,'\n');
fprintf(fid,'x0 : \t %0.4f\t %0.4f\t %0.4f\t %0.4f\t %0.4f\n', Others.x0);
fprintf(fid,'Perc. conv.: \t\t %0.4f', convperc);
fprintf(fid,'\n\n\n');

fclose(fid); 


n = size(DATA,1)-2;    


%PLOT THE DIAGNOSTIC TEST
diag_p = figure('position',[21    82   988   637],'name','Bootstrap Diagnosis','PaperOrientation', 'landscape', 'PaperPosition' , [0.25 0.25 10.5 8]);

disp(' ')
disp('Diagnostic:')
disp('     Horiz. dotted lines are mean +/- 3 S.D.')
disp(' ')


if (sum(DATA(1:n,1))) ~= n & (sum(DATA(1:n,1)) ~= 0),
    subplot(3,3,1)
    plot(1:size(obs,2),mean_check(:,1)-DATA(n+1,1),'*r')
    title('BIAS')
    ylabel('B*i - B*all')
    xrange = get(gca,'xlim');
    sd = 3.*std(mean_check(:,1)-DATA(n+1,1));
    center = mean(mean_check(:,1)-DATA(n+1,1));
    p_top = line(xrange,[center+sd center+sd]);
    p_bottom = line(xrange,[center-sd center-sd]);
    set(p_top,'color',[0 1 0],'LineStyle',':')
    set(p_bottom,'color',[0 1 0],'LineStyle',':')
end

if (sum(DATA(1:n,2))) ~= n & (sum(DATA(1:n,2)) ~= 0),
    subplot(3,3,4)
    plot(1:size(obs,2),mean_check(:,2)-DATA(n+1,2),'*r')
    eval(['title(''' model(1,:) ''')'])
    ylabel('B*i - B*all')
    sd = 3.*std(mean_check(:,2)-DATA(n+1,2));
    center = mean(mean_check(:,2)-DATA(n+1,2));
    p_top = line(xrange,[center+sd center+sd]);
    p_bottom = line(xrange,[center-sd center-sd]);
    set(p_top,'color',[0 1 0],'LineStyle',':')
    set(p_bottom,'color',[0 1 0],'LineStyle',':')
end

if (sum(DATA(1:n,3))) ~= n & (sum(DATA(1:n,3)) ~= 0),
    subplot(3,3,7)
    plot(1:size(obs,2),mean_check(:,3)-DATA(n+1,3),'*r')
    title(model(2,:))
    ylabel('B*i - B*all')
    sd = 3.*std(mean_check(:,3)-DATA(n+1,3));
    center = mean(mean_check(:,3)-DATA(n+1,3));
    p_top = line(xrange,[center+sd center+sd]);
    p_bottom = line(xrange,[center-sd center-sd]);
    set(p_top,'color',[0 1 0],'LineStyle',':')
    set(p_bottom,'color',[0 1 0],'LineStyle',':')
end

if (sum(DATA(1:n,4))) ~= n & (sum(DATA(1:n,4)) ~= 0),
    subplot(3,3,5)
    plot(1:size(obs,2),mean_check(:,4)-DATA(n+1,4),'*r')
    title(model(3,:))
    ylabel('B*i - B*all')
    sd = 3.*std(mean_check(:,4)-DATA(n+1,4));
    center = mean(mean_check(:,4)-DATA(n+1,4));
    p_top = line(xrange,[center+sd center+sd]);
    p_bottom = line(xrange,[center-sd center-sd]);
    set(p_top,'color',[0 1 0],'LineStyle',':')
    set(p_bottom,'color',[0 1 0],'LineStyle',':')
end

if (sum(DATA(1:n,5))) ~= n & (sum(DATA(1:n,5)) ~= 0),
    subplot(3,3,8)
    plot(1:size(obs,2),mean_check(:,5)-DATA(n+1,5),'*r')
    title(model(4,:))
    ylabel('B*i - B*all')
    sd = 3.*std(mean_check(:,5)-DATA(n+1,5));
    center = mean(mean_check(:,5)-DATA(n+1,5));
    p_top = line(xrange,[center+sd center+sd]);
    p_bottom = line(xrange,[center-sd center-sd]);
    set(p_top,'color',[0 1 0],'LineStyle',':')
    set(p_bottom,'color',[0 1 0],'LineStyle',':')
end

if (sum(DATA(1:n,11))) ~= n & (sum(DATA(1:n,11)) ~= 0),
    subplot(3,3,6)
    plot(1:size(obs,2),mean_check(:,11)-DATA(n+1,11),'*r')
    title('VAF')
    ylabel('B*i - B*all')
    sd = 3.*std(mean_check(:,11)-DATA(n+1,11));
    center = mean(mean_check(:,11)-DATA(n+1,11));
    p_top = line(xrange,[center+sd center+sd]);
    p_bottom = line(xrange,[center-sd center-sd]);
    set(p_top,'color',[0 1 0],'LineStyle',':')
    set(p_bottom,'color',[0 1 0],'LineStyle',':')
end

if (sum(DATA(1:n,12))) ~= n & (sum(DATA(1:n,12)) ~= 0),
    subplot(3,3,9)
    plot(1:size(obs,2),mean_check(:,12)-DATA(n+1,12),'*r')
    title('BIC')
    ylabel('B*i - B*all')
    sd = 3.*std(mean_check(:,12)-DATA(n+1,12));
    center = mean(mean_check(:,12)-DATA(n+1,12));
    p_top = line(xrange,[center+sd center+sd]);
    p_bottom = line(xrange,[center-sd center-sd]);
    set(p_top,'color',[0 1 0],'LineStyle',':')
    set(p_bottom,'color',[0 1 0],'LineStyle',':')
end

streamer('Bootstrap Diagnosis')


%PLOT THE PARAMETER DISTRIBUTIONS
disp(' ')
disp('Distributions:')
disp('     Solid red lines are estimates from the original data set')
disp('     Dotted green lines are normal approximations of a 95% two-sided confidence interval')
disp('         and are based on the S.E. of the estimates from the original data set')
disp('     Dotted yellow lines are bootstrap confidence intervals obtained with the BCa method')
disp(' ')

param_p = figure('position',[21    82   988   637],'name','Bootstrap Parameters','PaperOrientation', 'landscape', 'PaperPosition' , [0.25 0.25 10.5 8]);

if (sum(DATA(1:n,1))) ~= n & (sum(DATA(1:n,1)) ~= 0),
    subplot(3,3,1)
    hist(DATA(1:n,1),50);
    title(['BIAS;   Mean : ' num2str(DATA(n+1,1)) '; S.E. : ' num2str(DATA(n+2,1))])
    ylabel('N')
    yrange = get(gca,'ylim');
    p_mean = line([ORIG(1) ORIG(1)],yrange);
    p_left = line([ORIG(1)-z*ORIG(6) ORIG(1)-z*ORIG(6)],yrange);
    p_right = line([ORIG(1)+z*ORIG(6) ORIG(1)+z*ORIG(6)],yrange);
    set(p_mean,'color',[1 0 0],'LineStyle','-')
    set(p_left,'color',[0 1 0],'LineStyle',':')
    set(p_right,'color',[0 1 0],'LineStyle',':')
    p_bl = line([CI_BCa(1,1) CI_BCa(1,1)],yrange);
    p_br = line([CI_BCa(2,1) CI_BCa(2,1)],yrange);
    set(p_bl,'color',[1 1 0],'LineStyle',':')
    set(p_br,'color',[1 1 0],'LineStyle',':')
    
end

subplot(3,3,2)
axis off
text(0,1,['Parameter Estimates from Original Data Set:'])
if (sum(DATA(1:n,1))) ~= n & (sum(DATA(1:n,1)) ~= 0); text(0.1,0.8,['Bias +/- S.E. : ' num2str(ORIG(1)) ' , ' num2str(ORIG(6))]); end
if (sum(DATA(1:n,2))) ~= n & (sum(DATA(1:n,2)) ~= 0);text(0.1,0.65,[model(1,:) ' +/- S.E. : ' num2str(ORIG(2)) ' , ' num2str(ORIG(7))]); end
if (sum(DATA(1:n,3))) ~= n & (sum(DATA(1:n,3)) ~= 0);text(0.1,0.55,[model(2,:) ' +/- S.E. : ' num2str(ORIG(3)) ' , ' num2str(ORIG(8))]); end
if (sum(DATA(1:n,4))) ~= n & (sum(DATA(1:n,4)) ~= 0);text(0.1,0.4,[model(3,:) ' +/- S.E. : ' num2str(ORIG(4)) ' , ' num2str(ORIG(9))]); end
if (sum(DATA(1:n,5))) ~= n & (sum(DATA(1:n,5)) ~= 0);text(0.1,0.3,[model(4,:) ' +/- S.E. : ' num2str(ORIG(5)) ' , ' num2str(ORIG(10))]); end
if (sum(DATA(1:n,11))) ~= n & (sum(DATA(1:n,11)) ~= 0);text(0.1,0.15,['VAF : ' num2str(ORIG(11)) ]); end
if (sum(DATA(1:n,12))) ~= n & (sum(DATA(1:n,12)) ~= 0);text(0.1,0.05,['BIC : ' num2str(ORIG(12)) ]); end

subplot(3,3,3)
axis off
text(0,1,['Bootstrap Confidence Limits (BCa):'])
if (sum(DATA(1:n,1))) ~= n & (sum(DATA(1:n,1)) ~= 0); text(0.1,0.8,['Bias  : ' num2str(CI_BCa(:,1)')]); end
if (sum(DATA(1:n,2))) ~= n & (sum(DATA(1:n,2)) ~= 0); text(0.1,0.65,[model(1,:) '  : ' num2str(CI_BCa(:,2)')]); end
if (sum(DATA(1:n,3))) ~= n & (sum(DATA(1:n,3)) ~= 0); text(0.1,0.55,[model(2,:) '  : ' num2str(CI_BCa(:,3)')]); end
if (sum(DATA(1:n,4))) ~= n & (sum(DATA(1:n,4)) ~= 0); text(0.1,0.4,[model(3,:) '  : ' num2str(CI_BCa(:,4)')]); end
if (sum(DATA(1:n,5))) ~= n & (sum(DATA(1:n,5)) ~= 0); text(0.1,0.3,[model(4,:) '  : ' num2str(CI_BCa(:,5)')]); end
if (sum(DATA(1:n,11))) ~= n & (sum(DATA(1:n,11)) ~= 0); text(0.1,0.15,['VAF  : ' num2str(CI_BCa(:,11)')]); end
if (sum(DATA(1:n,12))) ~= n & (sum(DATA(1:n,12)) ~= 0); text(0.1,0.05,['BIC  : '  num2str(CI_BCa(:,12)')]); end

if (sum(DATA(1:n,2))) ~= n & (sum(DATA(1:n,2)) ~= 0),
    subplot(3,3,4)
    hist(DATA(1:n,2),50);
    title([model(1,:) ';   Mean : ' num2str(DATA(n+1,2)) '; S.E. : ' num2str(DATA(n+2,2))])
    ylabel('N')
    yrange = get(gca,'ylim');
    p_mean = line([ORIG(2) ORIG(2)],yrange);
    p_left = line([ORIG(2)-z*ORIG(7) ORIG(2)-z*ORIG(7)],yrange);
    p_right = line([ORIG(2)+z*ORIG(7) ORIG(2)+z*ORIG(7)],yrange);
    set(p_mean,'color',[1 0 0],'LineStyle','-')
    set(p_left,'color',[0 1 0],'LineStyle',':')
    set(p_right,'color',[0 1 0],'LineStyle',':')
    p_bl = line([CI_BCa(1,2) CI_BCa(1,2)],yrange);
    p_br = line([CI_BCa(2,2) CI_BCa(2,2)],yrange);
    set(p_bl,'color',[1 1 0],'LineStyle',':')
    set(p_br,'color',[1 1 0],'LineStyle',':')
end

if (sum(DATA(1:n,3))) ~= n & (sum(DATA(1:n,3)) ~= 0),
    subplot(3,3,7)
    hist(DATA(1:n,3),50);
    title([model(2,:) ';   Mean : ' num2str(DATA(n+1,3)) '; S.E. : ' num2str(DATA(n+2,3))])
    ylabel('N')
    yrange = get(gca,'ylim');
    p_mean = line([ORIG(3) ORIG(3)],yrange);
    p_left = line([ORIG(3)-z*ORIG(8) ORIG(3)-z*ORIG(8)],yrange);
    p_right = line([ORIG(3)+z*ORIG(8) ORIG(3)+z*ORIG(8)],yrange);
    set(p_mean,'color',[1 0 0],'LineStyle','-')
    set(p_left,'color',[0 1 0],'LineStyle',':')
    set(p_right,'color',[0 1 0],'LineStyle',':')
    p_bl = line([CI_BCa(1,3) CI_BCa(1,3)],yrange);
    p_br = line([CI_BCa(2,3) CI_BCa(2,3)],yrange);
    set(p_bl,'color',[1 1 0],'LineStyle',':')
    set(p_br,'color',[1 1 0],'LineStyle',':')
    
end

if (sum(DATA(1:n,4))) ~= n & (sum(DATA(1:n,4)) ~= 0),
    subplot(3,3,5)
    hist(DATA(1:n,4),50);
    title([model(3,:) ';   Mean : ' num2str(DATA(n+1,4)) '; S.E. : ' num2str(DATA(n+2,4))])
    ylabel('N')
    yrange = get(gca,'ylim');
    p_mean = line([ORIG(4) ORIG(4)],yrange);
    p_left = line([ORIG(4)-z*ORIG(9) ORIG(4)-z*ORIG(9)],yrange);
    p_right = line([ORIG(4)+z*ORIG(9) ORIG(4)+z*ORIG(9)],yrange);
    set(p_mean,'color',[1 0 0],'LineStyle','-')
    set(p_left,'color',[0 1 0],'LineStyle',':')
    set(p_right,'color',[0 1 0],'LineStyle',':')
    p_bl = line([CI_BCa(1,4) CI_BCa(1,4)],yrange);
    p_br = line([CI_BCa(2,4) CI_BCa(2,4)],yrange);
    set(p_bl,'color',[1 1 0],'LineStyle',':')
    set(p_br,'color',[1 1 0],'LineStyle',':')
    
end

if (sum(DATA(1:n,5))) ~= n & (sum(DATA(1:n,5)) ~= 0),
    subplot(3,3,8)
    hist(DATA(1:n,5),50);
    title([model(4,:) ';   Mean : ' num2str(DATA(n+1,5)) '; S.E. : ' num2str(DATA(n+2,5))])
    ylabel('N')
    yrange = get(gca,'ylim');
    p_mean = line([ORIG(5) ORIG(5)],yrange);
    p_left = line([ORIG(5)-z*ORIG(10) ORIG(5)-z*ORIG(10)],yrange);
    p_right = line([ORIG(5)+z*ORIG(10) ORIG(5)+z*ORIG(10)],yrange);
    set(p_mean,'color',[1 0 0],'LineStyle','-')
    set(p_left,'color',[0 1 0],'LineStyle',':')
    set(p_right,'color',[0 1 0],'LineStyle',':')
    p_bl = line([CI_BCa(1,5) CI_BCa(1,5)],yrange);
    p_br = line([CI_BCa(2,5) CI_BCa(2,5)],yrange);
    set(p_bl,'color',[1 1 0],'LineStyle',':')
    set(p_br,'color',[1 1 0],'LineStyle',':')
end

if (sum(DATA(1:n,11))) ~= n & (sum(DATA(1:n,11)) ~= 0),
    subplot(3,3,6)
    hist(DATA(1:n,11),50);
    title(['VAF;   Mean : ' num2str(DATA(n+1,11)) '; S.E. : ' num2str(DATA(n+2,11))])
    ylabel('N')
    yrange = get(gca,'ylim');
    p_mean = line([ORIG(11) ORIG(11)],yrange);
    set(p_mean,'color',[1 0 0],'LineStyle','-')
    p_bl = line([CI_BCa(1,11) CI_BCa(1,11)],yrange);
    p_br = line([CI_BCa(2,11) CI_BCa(2,11)],yrange);
    set(p_bl,'color',[1 1 0],'LineStyle',':')
    set(p_br,'color',[1 1 0],'LineStyle',':')
    
end

if (sum(DATA(1:n,12))) ~= n & (sum(DATA(1:n,12)) ~= 0),
    subplot(3,3,9)
    hist(DATA(1:n,12),50);
    title(['BIC;   Mean : ' num2str(DATA(n+1,12)) '; S.E. : ' num2str(DATA(n+2,12))])
    ylabel('N')
    yrange = get(gca,'ylim');
    p_mean = line([ORIG(12) ORIG(12)],yrange);
    set(p_mean,'color',[1 0 0],'LineStyle','-')
    p_bl = line([CI_BCa(1,12) CI_BCa(1,12)],yrange);
    p_br = line([CI_BCa(2,12) CI_BCa(2,12)],yrange);
    set(p_bl,'color',[1 1 0],'LineStyle',':')
    set(p_br,'color',[1 1 0],'LineStyle',':')
    
end

streamer('Bootstrap Parameters')


%PLOT THE CONFIDENCE INTERVALS
conf_p = figure('name','Confidence Intervals','position',[232   159   560   519]);

disp(' ')
disp(['Confidence intervals:'])
disp(['     Red: Bias, ' model(1,:) ' and ' model(2,:) ' from original data set'])
disp(['     Purple: ' model(3,:) ' and ' model(4,:) ' from original data set'])
disp(['     Green: Bias, ' model(1,:) ' ,' model(2,:) ' and VAF from bootstrap data set'])
disp(['     Yellow: ' model(3,:) ' , ' model(4,:) ' and BIC from bootstrap data set'])
disp(' ')

if (sum(DATA(1:n,1))) ~= n & (sum(DATA(1:n,1)) ~= 0),
    subplot(4,1,1)
    p_norm = line([ORIG(1)-z*ORIG(6) ORIG(1)+z*ORIG(6)],[2 2]);
    set(p_norm,'color',[1 0 0],'LineStyle','-','linewidth',10)
    text(ORIG(1)+z*ORIG(6)+0.5,2,'Bias_R')
    p_boot= line([CI_BCa(1,1) CI_BCa(2,1)],[1 1]);
    set(p_boot,'color',[0 1 0],'LineStyle','-','linewidth',10)
    text(CI_BCa(2,1)+0.5,1,'Bias_B')
    xrange = get(gca,'xlim');
    axis([xrange 0 3])
    ylabel('Bias')
end

subplot(4,1,2)
if (sum(DATA(1:n,2))) ~= n & (sum(DATA(1:n,2)) ~= 0),
    p_norm = line([ORIG(2)-z*ORIG(7) ORIG(2)+z*ORIG(7)],[5 5]);
    text(ORIG(2)+z*ORIG(7)+0.05,5,[model(1,:),'_1_R'])
    set(p_norm,'color',[1 0 0],'LineStyle','-','linewidth',10)
    p_boot= line([CI_BCa(1,2) CI_BCa(2,2)],[2 2]);
    text(CI_BCa(2,2)+0.05,2,[model(1,:),'_1_B'])
    set(p_boot,'color',[0 1 0],'LineStyle','-','linewidth',10)
end

if (sum(DATA(1:n,4))) ~= n & (sum(DATA(1:n,4)) ~= 0),
    p_norm1 = line([ORIG(4)-z*ORIG(9) ORIG(4)+z*ORIG(9)],[4 4]);
    set(p_norm1,'color',[1 0 0.5],'LineStyle','-','linewidth',10)
    text(ORIG(4)+z*ORIG(9)+0.05,4,[model(3,:),'_2_R'])
    p_boot1= line([CI_BCa(1,4) CI_BCa(2,4)],[1 1]);
    set(p_boot1,'color',[1 1 0],'LineStyle','-','linewidth',10)
    text(CI_BCa(2,4)+0.05,1,[model(3,:),'_2_B'])
end

xrange = get(gca,'xlim');
axis([xrange 0 6])
ylabel('Positions')

subplot(4,1,3)
if (sum(DATA(1:n,3))) ~= n & (sum(DATA(1:n,3)) ~= 0),
    p_norm = line([ORIG(3)-z*ORIG(8) ORIG(3)+z*ORIG(8)],[5 5]);
    set(p_norm,'color',[1 0 0],'LineStyle','-','linewidth',10)
    text(ORIG(3)+z*ORIG(8)+0.01,5,[model(2,:),'_1_R'])
    p_boot= line([CI_BCa(1,3) CI_BCa(2,3)],[2 2]);
    set(p_boot,'color',[0 1 0],'LineStyle','-','linewidth',10)
    text(CI_BCa(2,3)+0.01,2,[model(2,:),'_1_B'])
end

if (sum(DATA(1:n,5))) ~= n & (sum(DATA(1:n,5)) ~= 0),
    p_norm = line([ORIG(5)-z*ORIG(10) ORIG(5)+z*ORIG(10)],[4 4]);
    set(p_norm,'color',[1 0 0.5],'LineStyle','-','linewidth',10)
    text(ORIG(5)+z*ORIG(10)+0.01,4,[model(4,:),'_2_R'])
    p_boot1= line([CI_BCa(1,5) CI_BCa(2,5)],[1 1]);
    set(p_boot1,'color',[1 1 0],'LineStyle','-','linewidth',10)
    text(CI_BCa(2,5)+0.01,1,[model(4,:),'_2_B'])
end

xrange = get(gca,'xlim');
axis([xrange 0 6])
ylabel('Velocities')

subplot(4,1,4)
if (sum(DATA(1:n,11))) ~= n & (sum(DATA(1:n,11)) ~= 0),
    p_VAF= line([CI_BCa(1,11) CI_BCa(2,11)],[2 2]);
    set(p_VAF,'color',[1 0 0],'LineStyle','-','linewidth',10)
    text(CI_BCa(2,11)+0.05,2,'VAF')
end

if (sum(DATA(1:n,12))) ~= n & (sum(DATA(1:n,12)) ~= 0),
    p_BIC= line([CI_BCa(1,12) CI_BCa(2,12)],[1 1]);
    set(p_BIC,'color',[0 1 0],'LineStyle','-','linewidth',10)
    text(CI_BCa(2,12)+0.05,1,'BIC')
end

xrange = get(gca,'xlim');
axis([xrange 0 3])
ylabel('VAF & BIC')

streamer('Confidence Intervals')



%SUMMARY DIAGRAM
summary_p = figure('position',[96   243   889   435],'name','Summary diagram','PaperOrientation', 'landscape', 'PaperPosition' , [0.25 0.25 10.5 8]);

%positions (#1 & 3)
if (model(1,:) ~= '---') & (model(3,:) ~= '---'),
    if ((model(2,:) ~= '---') & (model(4,:) ~= '---'))
        subplot(1,2,1)
    end
    hist(DATA(1:n,[2 4]),50);
    title([model(1,:) '_1 (blue):  ' num2str(DATA(n+1,2),3) ' +/- ' num2str(DATA(n+2,2),3) ' ; ' model(3,:) '_2 (red):  ' num2str(DATA(n+1,4),3) ' +/- ' num2str(DATA(n+2,4),3)])
    ylabel('N')
    xlabel('Parameter value')
    yrange = get(gca,'ylim');
    
    p_mean = line([ORIG(2) ORIG(2)],yrange);
    set(p_mean,'color',[1 0 0],'LineStyle','-')
    p_boot= line([CI_BCa(1,2) CI_BCa(2,2)],[-15 -15]);
    text(CI_BCa(2,2)+0.05,-15,[model(1,:),'_1'])
    set(p_boot,'color',[0 0 1],'LineStyle','-','linewidth',10)
    
    p_mean = line([ORIG(4) ORIG(4)],yrange);
    set(p_mean,'color',[1 0 0],'LineStyle','-')
    p_boot1= line([CI_BCa(1,4) CI_BCa(2,4)],[-30 -30]);
    set(p_boot1,'color',[1 0 0],'LineStyle','-','linewidth',10)
    text(CI_BCa(2,4)+0.05,-30,[model(3,:),'_2'])
    
    axis tight
    axis([xlim -40 max(ylim)]) 
    
elseif (model(1,:) ~= '---'),
    if ((model(2,:) ~= '---') & (model(4,:) ~= '---'))
        subplot(1,2,1)
    end
    hist(DATA(1:n,[2]),50);
    title([model(1,:) '_1 (blue):  ' num2str(DATA(n+1,2),3) ' +/- ' num2str(DATA(n+2,2),3) ' ;'])
    ylabel('N')
    xlabel('Parameter value')
    yrange = get(gca,'ylim');
    
    p_mean = line([ORIG(2) ORIG(2)],yrange);
    set(p_mean,'color',[1 0 0],'LineStyle','-')
    p_boot= line([CI_BCa(1,2) CI_BCa(2,2)],[-15 -15]);
    text(CI_BCa(2,2)+0.05,-15,[model(1,:),'_1'])
    set(p_boot,'color',[0 0 1],'LineStyle','-','linewidth',10)
    
    axis tight
    axis([xlim -40 max(ylim)]) 
    
elseif (model(3,:) ~= '---'),
    if ((model(2,:) ~= '---') & (model(4,:) ~= '---'))
        subplot(1,2,1)
    end
    hist(DATA(1:n,[4]),50);
    title([model(3,:) '_2 (blue):  ' num2str(DATA(n+1,4),3) ' +/- ' num2str(DATA(n+2,4),3) ' ;'])
    ylabel('N')
    xlabel('Parameter value')
    yrange = get(gca,'ylim');
    
    p_mean = line([ORIG(4) ORIG(4)],yrange);
    set(p_mean,'color',[1 0 0],'LineStyle','-')
    p_boot1= line([CI_BCa(1,4) CI_BCa(2,4)],[-30 -30]);
    set(p_boot1,'color',[1 0 0],'LineStyle','-','linewidth',10)
    text(CI_BCa(2,4)+0.05,-30,[model(3,:),'_2'])
    
    axis tight
    axis([xlim -40 max(ylim)]) 
    
end


%velocities (#2 & 4)
if ((model(2,:) ~= '---') & (model(4,:) ~= '---'))
    if (model(2,:) ~= '---') & (model(4,:) ~= '---'),
        subplot(1,2,2)
        hist(DATA(1:n,[3 5]),50);
        title([model(2,:) '_1 (blue):  ' num2str(DATA(n+1,3),3) ' +/- ' num2str(DATA(n+2,3),3) ' ; ' model(4,:) '_2 (red):  ' num2str(DATA(n+1,5),3) ' +/- ' num2str(DATA(n+2,5),3)])
        ylabel('N')
        xlabel('Parameter value')
        yrange = get(gca,'ylim');
        
        p_mean = line([ORIG(3) ORIG(3)],yrange);
        set(p_mean,'color',[1 0 0],'LineStyle','-')
        p_boot= line([CI_BCa(1,3) CI_BCa(2,3)],[-15 -15]);
        set(p_boot,'color',[0 0 1],'LineStyle','-','linewidth',10)
        text(CI_BCa(2,3)+0.01,-15,[model(2,:),'_1'])
        
        p_mean = line([ORIG(5) ORIG(5)],yrange);
        set(p_mean,'color',[1 0 0],'LineStyle','-')
        p_boot1= line([CI_BCa(1,5) CI_BCa(2,5)],[-30 -30]);
        set(p_boot1,'color',[1 0 0],'LineStyle','-','linewidth',10)
        text(CI_BCa(2,5)+0.01,-30,[model(4,:),'_2'])
        
        axis tight
        axis([xlim -40 max(ylim)]) 
        
    elseif (model(2,:) ~= '---'),
        subplot(1,2,2)
        hist(DATA(1:n,[3]),50);
        title([model(2,:) '_1 (blue):  ' num2str(DATA(n+1,3),3) ' +/- ' num2str(DATA(n+2,3),3) ' ;'])
        ylabel('N')
        xlabel('Parameter value')
        yrange = get(gca,'ylim');
        
        p_mean = line([ORIG(3) ORIG(3)],yrange);
        set(p_mean,'color',[1 0 0],'LineStyle','-')
        p_boot= line([CI_BCa(1,3) CI_BCa(2,3)],[-15 -15]);
        set(p_boot,'color',[0 0 1],'LineStyle','-','linewidth',10)
        text(CI_BCa(2,3)+0.01,-15,[model(2,:),'_1'])
        
        axis tight
        axis([xlim -40 max(ylim)]) 
        
    elseif (model(4,:) ~= '---'),
        subplot(1,2,2)
        hist(DATA(1:n,[5]),50);
        title([model(4,:) '_2 (blue):  ' num2str(DATA(n+1,5),3) ' +/- ' num2str(DATA(n+2,5),3) ' ;'])
        ylabel('N')
        xlabel('Parameter value')
        yrange = get(gca,'ylim');
        
        p_mean = line([ORIG(5) ORIG(5)],yrange);
        set(p_mean,'color',[1 0 0],'LineStyle','-')
        p_boot1= line([CI_BCa(1,5) CI_BCa(2,5)],[-30 -30]);
        set(p_boot1,'color',[1 0 0],'LineStyle','-','linewidth',10)
        text(CI_BCa(2,5)+0.01,-30,[model(4,:),'_2'])
        
        axis tight
        axis([xlim -40 max(ylim)]) 
        
    end
end



%SAVE AND PRINT OUTPUT FIGURES
filename_diag = [filename_m(1:length(filename_m)-4) '_diag.fig'];
filename_param = [filename_m(1:length(filename_m)-4) '_param.fig'];
filename_conf = [filename_m(1:length(filename_m)-4) '_conf.fig'];
filename_sum = [filename_m(1:length(filename_m)-4) '_sum.fig'];
filename_txt = [filename_m(1:length(filename_m)-4) '_summary.txt'];

saveas(diag_p,filename_diag)
saveas(param_p,filename_param)
saveas(conf_p,filename_conf)
saveas(summary_p,filename_sum)

if (print_flag == 1)
    print(diag_p)
    print(param_p)
    print(conf_p)
    print(summary_p)
end

%close(diag_p,param_p,conf_p,summary_p)
close(param_p,conf_p)

edit(filename_txt)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  SUBFUNCTION #1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [CI_BCa] = BS_conf(ORIG,DATA,M,ns,lat,out1,in1to4,mode_flag,userdir,alpha)

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
    [x,sd,error_sim,VAF,BIC,est,Others] = dodyn3(M,ns,lat,out1,in1to4,mode_flag);
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

%SKIP THE STANDARD ERRORS, and go directly to the VAF, BIC, and sum of residual error

for i = 11:12,
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







%%%%%%%%%%%%%%%%%%%%%%%%%5
% SUBFUNCTION #2
%%%%%%%%%%%%%%%%%%%%%%%%%

function [mean_check] = BS_diagnose(DATA,obs,ns)

mean_check = zeros(size(obs,2),size(DATA,2));

for i = 1:size(obs,2),
    idf_t = find(sum(obs' == ns(i)) == 0);
    if ~isempty(idf_t)
        if (size(idf_t,2) > 1),
            mean_check(i,:) = mean(DATA(idf_t,:));
        else
            mean_check(i,:) = (DATA(idf_t,:));
        end      
    else
        mean_check(i,:) = NaN;
    end
end





