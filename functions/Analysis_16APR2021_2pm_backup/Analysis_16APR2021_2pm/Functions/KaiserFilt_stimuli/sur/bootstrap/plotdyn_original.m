%PAS, 2000

n_imp = Others.n_imp;

% disp(' ');
disp('______________________________________');
disp(['Number of saccades tested: ', int2str(size(ns,2)),';  Number of data points: ',num2str(Others.NN),';   Lead Time: ',num2str(lat)]);
% disp(' ')
disp('MODEL:')
if (model(5,1) ~= 'B'),
    disp(sprintf('  %s = bias + %s + %s + %s + %s\n', model(5,:),model(1,:),model(2,:),model(3,:),model(4,:) ));
else
    disp(sprintf('  %s = ---- + %s + %s + %s + %s\n', model(5,:),model(1,:),model(2,:),model(3,:),model(4,:) ));
end
% disp(' ')
disp(sprintf('Variable\t\t Gain\t S.E.\t'))
% disp(sprintf('Variable\t\t Gain\t S.E.\t 95%% LB\t 95%% RB'))
% disp(' ')

%Set double sided confidence limit
alpha = 0.95; 
Z = []; Z = -norminv(0.05,0,1);

CL = zeros(5,2);

if (se(1) ~= 0),
    CL(1,1) = x(1) - Z*se(1);
    CL(1,2) = x(1) + Z*se(1);
else
    CL(1,:) = [0 0];
end

if (x(1) ~= 0),
     disp(sprintf('Bias : \t\t %0.4f\t %0.4f\t ', x(1), se(1) )); 
%     disp(sprintf('Bias : \t\t %0.4f\t %0.4f\t %0.4f\t %0.4f', x(1) , se(1) , CL(1,1) , CL(1,2) )); 
end

for i = 2:size(n_imp,2)+1,
    if (se(i) ~= 0),
        CL(i,1) = x(i) - Z*se(i);
        CL(i,2) = x(i) + Z*se(i);
    else
        CL(i,:) = [0 0];
    end
    
    if (model(i-1,:) == '---'),
    else
        disp(sprintf('%s : \t\t %0.4f\t %0.4f\t ', model(i-1,:) , x(i), se(i) )); 
%         disp(sprintf('%s : \t\t %0.4f\t %0.4f\t %0.4f\t %0.4f', model(i-1,:) , x(i) , se(i) , CL(i,1) , CL(i,2) )); 
disp(sprintf('  %s = bias + %s%0.4f + %s%0.4f + %s%0.4f + %s%0.4f\n', model(5,:),model(1,:),x(1),model(2,:),x(2),model(3,:),x(3),model(4,:),x(4) ));
    end
end

% disp(' ')
% disp(sprintf('Mean Error : \t\t%0.4f',error_sim)); trying something
disp(sprintf('VAF : \t\t%0.4f',VAF)); 
% disp(sprintf('BIC : \t\t%0.4f',BIC)); 
% disp(' ')
% disp(sprintf('x0 : \t\t %0.4f\t %0.4f\t %0.4f\t %0.4f\t %0.4f', Others.x0));
% disp(sprintf('Convergence status : \t\t %0.4f', Others.exit));
% disp(sprintf('Iteration time : \t\t %0.4f', Others.timeit));
clear Z i j

sum_vect = [];
sum_vect = [length(est)];
for i = 1:5
    sum_vect = [sum_vect x(i) se(i) CL(i,1) CL(i,2)];  
end
sum_vect = [sum_vect VAF BIC error_sim];
%disp('Summary:')
%disp(sprintf('%0.6f\t',sum_vect))


%Plot the results
if(1)
    
    %n = size(ns,2);
    %Mi=zeros(n,2);  %set SIZE of Mi equal to size Md. 
    %for i = 1:n
    %    Mi(i,1)=M(ns(i),1); 
    %    Mi(i,2)=M(ns(i),2);	%set values of Mi markers equal to M markers, which gets rid of the buffer
    %end 
    
    %neg_val = find(est(:,:) < 0);
    %est(neg_val) = 0;
    
    %plot_vector = ones(size(fr)).*x(1);
    %start = 1;
    %for i = 1:size(Md,1),
    %    seg = (Mi((i),1)+lat):(Mi((i),2)+lat);
    %    end_seg = start + length(seg) - 1;
    %    plot_vector(seg) = est(start:end_seg);
    %    start = start + length(seg);
    %end;
    
    plot_vector = fr;   
    plot_vector(:,:) = NaN;
    if ( (Plotlist(m-1,3) == 'v') | (Plotlist(m-1,1:3) == 'vcj') | (Plotlist(m-1,1:3) == 'vvg') );
        %    plot_vector(:,:) = 0;
        plot_vector(Others.time_array+lat-1) = est;
    else
        plot_vector(Others.time_array+lat) = est;
    end
    temp_vtar = vtar;
    vtar = [];
    vtar = plot_vector;
    evalin('base','clear model_fit;')
    assignin('base','model_fit',plot_vector);
    
    Plotlist = str2mat(Plotlist,'vtr');
    NoofDisplayed = length( Plotlist( :, 1));
    plotstr = plotmate( Plotlist, Signals, Signaldefinitions);
    displaytitle = [dataname ': ']; 
    for i = 1:NoofDisplayed, displaytitle = [displaytitle lower(Plotlist(i,1:3)) ' '];end
    replot;
    vtar = temp_vtar;
    Plotlist(length(Plotlist),:) = [];
    NoofDisplayed = length( Plotlist( :, 1));
    plotstr = plotmate( Plotlist, Signals, Signaldefinitions);
    displaytitle = [dataname ': ']; 
    for i = 1:NoofDisplayed, displaytitle = [displaytitle lower(Plotlist(i,1:3)) ' '];end
    
end

clear n_imp clear NN start seg end_seg plot_vector neg_val temp_vtar
clear x se error_sim VAF BIC est Others
clear global t_in t_out x
