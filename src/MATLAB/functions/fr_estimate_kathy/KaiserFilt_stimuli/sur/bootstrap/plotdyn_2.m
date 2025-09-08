%PAS, 2000

n_imp = Others.n_imp;

% disp(' ');
disp('______________________________________');
disp(sprintf('Number of saccades tested:\t %d\t Number of data points:\t %d\t Lead Time:\t %d', size(ns,2), Others.NN, lat));
%disp(['Number of saccades tested: ', int2str(size(ns,2)),';  Number of data points: ',num2str(Others.NN),';   Lead Time: ',num2str(lat)]);
% disp(' ')
disp('MODEL:')
if (model(5,1) ~= 'B'),
     disp(sprintf('  %s = bias + %s + %s + %s + %s\n', model(5,:),model(1,:),model(2,:),model(3,:),model(4,:) ));
else
     disp(sprintf('  %s = ---- + %s + %s + %s + %s\n', model(5,:),model(1,:),model(2,:),model(3,:),model(4,:) ));
end
 disp(' ')
% disp(sprintf('Variable\t\t Gain\t S.E.\t'))
 disp(sprintf('Variable\t Gain\t S.E.\t 95%% LB\t 95%% RB'))
 disp(' ')

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
  %    disp(sprintf('Bias : \t\t %0.4f\t %0.4f\t ', x(1), se(1) )); 
     disp(sprintf('Bias : \t\t %0.4f\t %0.4f\t %0.4f\t %0.4f', x(1) , se(1) , CL(1,1) , CL(1,2) )); 
end


for i = 2:size(n_imp,2)+1,
    if (se(i) ~= 0),
        CL(i,1) = x(i) - Z*se(i);
        CL(i,2) = x(i) + Z*se(i);
    else
        CL(i,:) = [0 0];
    end
    
    if strcmp(model(i-1,:),'---'),
    else
       %  disp(sprintf('%s : \t\t %0.4f\t %0.4f\t ', model(i-1,:) , x(i), se(i) )); 
         disp(sprintf('%s : \t\t %0.4f\t %0.4f\t %0.4f\t %0.4f', model(i-1,:) , x(i) , se(i) , CL(i,1) , CL(i,2) )); 
end
end

 disp(' ')
% disp(sprintf('Mean Error : \t\t%0.4f',error_sim)); trying something
 disp(sprintf('VAF : \t\t%0.4f',VAF)); 
 disp(sprintf('BIC : \t\t%0.4f',BIC)); 
 disp(' ')
% disp(sprintf('x0 : \t\t %0.4f\t %0.4f\t %0.4f\t %0.4f\t %0.4f', Others.x0));
 %disp(sprintf('Convergence status : \t\t %0.4f', Others.exit));
 %disp(sprintf('Iteration time : \t\t %0.4f', Others.timeit));
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
if(plotflag)
    
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
    
    %plot_vector = fr;  
    plot_vector = zeros(N,1);
    plot_vector(:,:) = NaN;

    plot_vector(Others.time_array+lat) = est;

    newname = inputdlg('Name your filtered signal here:','New Name', 1,{'fr_est'},'on');
    newname = strtrim(cell2mat(newname));
    
    if isempty(newname)
        errordlg('Filtered signal name blank.');
        error('Filtered signal name blank.')
    end

    Data.(newname) = plot_vector;
    evalin('base','clear model_fit;')
    assignin('base','model_fit',plot_vector);
    
    if (isempty(strmatch(newname, Plotlist)))
            Plotlist_back = str2mat(Plotlist_back, strtrim(newname));
            Signals_back = str2mat(Signals_back,strtrim(newname));
            Signaldefinitions_back = str2mat(Signaldefinitions_back, ['Data.',strtrim(newname),'(1+lat:N)']);
            Signaldescriptions = str2mat(Signaldescriptions, newname);

            viewArray(end+1)=1;
            viewArrayL=logical(viewArray);

            NoofDisplayed = length( Plotlist( :, 1));
            %Data.Definitions{NoofDisplayed} = ['Data.',strtrim(NewName),'(1+lat:N)'];
            DefinitionField = ['Data.',strtrim(newname),'(1+lat:N)'];


    end

    if ~sum(find(strcmp(newname,Data.ChannelList)))
        Data.ChannelList(end+1)={newname};

%         Data.ChannelNames(end+1,:)='                ';
%         Data.ChannelNames(end,1:length(newname))=newname;

        Data.ChannelNumbers(end+1)=0;
        Data.ChNumbers(end+1)={'0'};

        Data.adfreq(end+1)=1000;
        Data.samples(end+1)=Data.samples(1);
        Data.SampleCounts=Data.samples;                       
        Data.NumberOfChannels=length(Data.ChannelList);
        Data.NumberOfSignals=length(Data.ChannelList);
        Data.Definitions(Data.NumberOfSignals)={['Data.' newname '(1+lat:N)']};
    end

    signal_num=signal_num+1;
    
    Plotlist=Plotlist_back(viewArrayL,:);
    noPlotlist=Plotlist_back(~viewArrayL,:);

    Signals=Signals_back(viewArrayL,:);
    Signaldefinitions=Signaldefinitions_back(viewArrayL,:);


    set(show_list,'String',Plotlist)
    if isempty(get(show_list,'value'))
        set(show_list,'value',1)
    elseif get(show_list,'value')>size(Plotlist,1) || get(show_list,'value')<1
        set(show_list,'value',1)
    end
    set(no_show_list,'String',noPlotlist)
    if isempty(get(no_show_list,'value'))
        set(no_show_list,'value',1)
    elseif get(no_show_list,'value')>size(noPlotlist,1) || logical(get(no_show_list,'value')<1)
        set(no_show_list,'value',1)
    end


    replot1;
    %vtar = temp_vtar;
    %Plotlist(length(Plotlist),:) = [];
    %NoofDisplayed = length( Plotlist( :, 1));
    %plotstr = plotmate( Plotlist, Signals, Signaldefinitions);
    %displaytitle = [dataname ': ']; 
    %for i = 1:NoofDisplayed, displaytitle = [displaytitle lower(Plotlist(i,1:3)) ' '];end
    
end






disp(sprintf('%s=\t%0.4g\t+%s*\t%0.4g\t+%s*\t%0.4g\t+%s*\t%0.4g\t+%s*\t%0.4g\tVAF=\t%0.4g\n', model(5,:),x(1),model(1,:),x(2),model(2,:),x(3),model(3,:),x(4),model(4,:),x(5), VAF ));
clear n_imp clear NN start seg end_seg plot_vector neg_val temp_vtar
clear x se error_sim VAF BIC est Others
clear global t_in t_out x
