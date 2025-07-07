function [filename_m] = bootstrap(M,ns,lat,out1,in1to4,mode_flag,userdir,model,resume_flag,multi_name)

%Bootstrap analysis as per Carpenter and Bithell 2000
%
%Must be used through dynatwo.m to insure proper formatting of the inputs
%
%Calls dodyn3.m for actual fitting, and postboot.m for 'post-oc' analysis
%
%PAS, 2000

global abort_flag

abort_flag = 0;
if (nargin < 10),
    multi_name = [];
end


x0 = [100 0 0 0 0];  %default initial conditions

if (size(mode_flag,2) > 1) & (mode_flag(1) == 8888)  %for multi-units
    mode_flag = mode_flag;
elseif (size(mode_flag,2) > 1) & (mode_flag(2) == 5678)  %for multi-units
    mode_flag = mode_flag;
elseif (mode_flag == 9999),         %in the case where you run a single bootstrap run
    mode_flag = [8888 getini(x0)]; 
elseif (mode_flag == 1234),         %in the case where you run a single bootstrap run
    mode_flag = mode_flag; 
else
    mode_flag = 0;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  SET/GET SOME IMPORTANT VARIABLES

n_seg = size(ns,2);
ns_ini = ns;
ORIG = []; DATA = []; obs = []; time = [];

rand('state',sum(100*clock));

ite = 1999;        %number of iterations to be performed
disp_ite = 50;     %frequency of result display in workspace
save_ite = 100;    %frequency of "safety" saves

post_flag = 1;     %automatically perform post-bootstrap analysis
print_flag = 0;    %print the analysis results

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


disp(' ')
disp('_______________________________________')
disp('Computing Bootstrap Algorithm ...')
disp('     Click on Progress Bar to ABORT and SAVE the current results')
disp(' ')



%Set/Get the proper file info
id = 1;
if (resume_flag == 0),
    %ESTIMATE VALUES FROM ORIGINAL DATA SET
    if (size(mode_flag,2) > 1) & (mode_flag(2) == 5678),
        [x,sd,error_sim,VAF,BIC,est,Others] = dynlatpick(M,ns_ini,lat,out1,in1to4,mode_flag);
        ORIG = [x sd VAF BIC error_sim Others.x0 Others.exit Others.lat];
    else 
        [x,sd,error_sim,VAF,BIC,est,Others] = dodyn3(M,ns_ini,lat,out1,in1to4,mode_flag);
        ORIG = [x sd VAF BIC error_sim Others.x0 Others.exit];
    end
    
    q = 1;   
    if (isempty(multi_name) == 1),
        [fname,pname] = uiputfile('*.mat','Enter Filename (*.MAT)'); 
        filename_e = [pname fname '.mat']; 
        userdir = pname;  
        save(filename_e,'M','ns','lat','out1','in1to4','userdir','model','mode_flag'); 
        dt = fname;
    else
        dt = multi_name;
    end
    
    filename = [cd '\bs_' dt '_' num2str(id)];
    while (exist([filename '.txt'],'file') == 2),
        id = id+1;
        filename = [cd '\bs_' dt '_' num2str(id)];
    end
    filename_t = [filename '.txt'];
    filename_m = [filename '.mat'];
    
    disp(' ')
    disp('CURRENT MODEL:')
    disp(sprintf('  %s = bias + %s + %s + %s + %s\n', model(5,:),model(1,:),model(2,:),model(3,:),model(4,:) ));
    disp(' ')
    
else
    %LOAD ORIGINAL DATA FILE
    if (isempty(multi_name) == 1),
        [fname,pname] = uigetfile('*.mat','Select *.MAT Data File to be continued');
        model_pick = model;
        load([pname fname])
        disp('     Previous iterations loaded')
        disp(' ')
        
        if (sum(sum(model_pick == model)) ~= 15),
            disp(' ')
            disp('!!! WARNING: The current inputs do NOT match the saved model !!!')
            disp('!!! Process was aborted  !!!')
            disp(' ')
            disp('CURRENT MODEL:')
            disp(sprintf('  %s = bias + %s + %s + %s + %s\n', model_pick(5,:),model_pick(1,:),model_pick(2,:),model_pick(3,:),model_pick(4,:) ));
            disp('SAVED MODEL:')
            disp(sprintf('  %s = bias + %s + %s + %s + %s\n', model(5,:),model(1,:),model(2,:),model(3,:),model(4,:) ));
            disp(' ')
            abort_flag = 2;
        end;
        
        DATA =  DATA(1:size(DATA,1)-2,:);
        
        q = size(DATA,1)+1;
        
        filename_m = [cd '\' fname];
        filename_t = [cd '\' fname(1:length(fname)-3) 'txt'];
        
    else  %resume multi-unit
        
        q = 1;   
        dt = multi_name;
        id = 1;
        
        filename = [cd '\bs_' dt '_' num2str(id)];
        
        filename_t = [filename '.txt'];
        filename_m = [filename '.mat'];
        
        
        load([filename_m])
        disp('     Previous iterations loaded')
        disp(' ')
        
        DATA =  DATA(1:size(DATA,1)-2,:);
        
        q = size(DATA,1)+1;
        
        disp(' ')
        disp('CURRENT MODEL:')
        disp(sprintf('  %s = bias + %s + %s + %s + %s\n', model(5,:),model(1,:),model(2,:),model(3,:),model(4,:) ));
        disp(' ')
        
    end
    
end


%setup the abort option
h = waitbar(0,'Please wait...');
set(h,'WindowButtonDownFcn','abort_flag = 1;','color',[.2 .5 .2],'menubar','none','name','Click Progress Bar to ABORT and SAVE...','number','off','position',[264.7500  417.0000  270.0000   56.2500]);
pause(2)


%MAIN BOOTSTRAP LOOP
while (q <= ite) & (abort_flag == 0),
    %ESTIMATE

    t0 = clock;
    obs(q,:) = ns_ini(ceil(rand(1,n_seg) .* n_seg));
    
    if (size(mode_flag,2) > 1) & (mode_flag(2) == 5678),
        [x,se,error_sim,VAF,BIC,est,Others] = dynlatpick(M,obs(q,:),lat,out1,in1to4,mode_flag);
        DATA(q,:) = [x se VAF BIC error_sim Others.x0 Others.exit Others.lat];
    else,
        [x,se,error_sim,VAF,BIC,est,Others] = dodyn3(M,obs(q,:),lat,out1,in1to4,mode_flag);
        DATA(q,:) = [x se VAF BIC error_sim Others.x0 Others.exit];
    end
    
    time(q) = etime(clock,t0);
    
    %Display once in a while & save for safety
    if (mod(q,disp_ite) == 0),
        waitbar(q/ite,h)
        t_left = ( ( mean(time) .* ite ) - sum(time) );
        disp(['Iteration #' num2str(q) ' of ' num2str(ite) '; Avg Iteration: ' num2str(mean(time(q-disp_ite+1:q)),2) ' sec; Estimated Time Remaining: ' num2str(floor(t_left/3600)) 'h' num2str(floor((t_left-floor(t_left/3600)*3600)/60)) 'min' num2str(floor(mod(t_left,60))) 'sec'])
    end
    
    if (mod(q,save_ite) == 0),
        save_it(model,DATA,ORIG,obs,time,filename_m,filename_t,1,mode_flag,Others)
        disp('   Saved Partial Results (*.mat) ...')
    end
    
    q = q + 1;
end

close(h) 


%Message if aborted
if (abort_flag == 1),
    disp('User aborted ...')
    post_flag = 0;
end


%Do final stuff
if (abort_flag ~= 2),
    
    save_it(model,DATA,ORIG,obs,time,filename_m,filename_t,2,mode_flag,Others)
    
    %Display in workspace
    disp(' ')
    disp('DONE ...')
    disp('     Final data was saved (*.mat and *.txt) ...')
    disp(' ')
    
    %Call Post-Bootstrap analysis package
    if (post_flag == 1),
        postboot(filename_m,M,ns,lat,out1,in1to4,userdir,Others,print_flag);
    end
end


clear global abort_flag



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     SAVE ROUTINE SUB-FUNCTION     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = save_it(model,DATA,ORIG,obs,time,filename_m,filename_t,txt_flag,mode_flag,Others)

%COMPUTE MEAN AND SD, AND HAPPEND AT THE END OF DATA
DATA = [DATA ; mean(DATA) ; std(DATA)];
dur = sum(time);
convperc = 100*(size(find(DATA(1:size(DATA,1)-2,19) > 0),1) / (size(DATA,1)-2));

%Save to matlab
save(filename_m,'filename_m','filename_t','model','ORIG','DATA','obs','time','mode_flag','Others','convperc')

%Save to text
if (txt_flag == 2),
    fid = fopen(filename_t,'wt'); 
    fprintf(fid,'__________________________________________________________________\n');
    fprintf(fid,'MODEL:  \n\n');
    fprintf(fid,'%s = bias + %s + %s + %s + %s\n', model(5,:),model(1,:),model(2,:),model(3,:),model(4,:) );
    
    fprintf(fid,'\n\n\n');
    fprintf(fid,'__________________________________________________________________\n');
    fprintf(fid,'Original Data Set Estimates:  (parameters, s.e., VAF, BIC, error_sim)\n\n');
    for k = 1:size(ORIG,2),
        fprintf(fid,'%0.4f\t', ORIG(1,k));
    end
    fprintf(fid,'\n\n\n\n\n');
    
    fprintf(fid,'__________________________________________________________________\n');
    fprintf(fid,'Bootstrap Iterations:  (parameters, s.e., VAF, BIC, error_sim)\n\n');
    for j = 1:size(DATA,1);
        if (j == size(DATA,1)-1),
            fprintf(fid,'\n');
            fprintf(fid,'AVERAGE and STANDARD ERROR: \n');
        end
        fprintf(fid,'#%i\t', j);
        for k = 1:size(DATA,2),
            fprintf(fid,'%0.4f\t', DATA(j,k));
        end
        fprintf(fid,'\n');
    end
    fprintf(fid,'\nTime (s):\t %0.4f\t # Iter.:\t %0.4f\t # Sacc.:\t %0.4f\n\n', dur,size(DATA,1)-2, size(obs,2));
    %fprintf(fid,'Percent Converged trials:\t\t\t %0.4f\n\n', convperc);
    %fprintf(fid,'Initial Conditions:\t %0.4f\t %0.4f\t %0.4f\t %0.4f\t %0.4f\n\n\n\n\n', Others.x0(:,:));
    
    fprintf(fid,'__________________________________________________________________\n');
    fprintf(fid,'Observations:\n');
    for j = 1:size(obs,1);
        for k = 1:size(obs,2),
            obs1 = sort(obs(j,:));
            fprintf(fid,'%i\t', obs1(1,k));
        end
        fprintf(fid,'\n');
    end
    
    fclose(fid); 
end
%END OF SAVE_IT


