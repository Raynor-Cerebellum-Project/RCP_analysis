function [fname_d] = bs_multi(choice)

global abort_flag


%First get directory info

%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%
%SET SOME OPTIONS


verg_flag = 1;         %Used to convert binocular model to conjugate/vergence model
optimize_flag = 0;     %Used to "optimize" the initial conditions in order to speed up the bootstrap
email_flag = 0;        %Used to "email" me when a cell is completed
BN_flag = 1;
email_address = 'pa.sylvestre@mail.mcgill.ca';     
current_computer = 'schumi';

%%%%%%%%%%%%%%%%%%%%%%%%%

t0 = [];
t0 = clock;


if (nargin == 0),
    choice = 0;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%DO restarted file first
%Skip for most options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (choice == 1),
    
    clc
    disp(' ')
    disp('Make sure that the FIRST file you pick')
    disp('is the file that was interrupted!')
    disp(' ')
    
    [filename_cont, pathname] = uigetfile('*.mat', 'Pick a data file');
    
    disp('NOTE: Picked file to be restarted!')
    disp(' ')
    
    disp(['Current RE-STARTED file: ' filename_cont])
    disp(' ')
    
    clear M ns lat out1 in1to4 userdir model mode_flag
    load(filename_cont)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %IF required, build conjugate and vergence model
    
        if (verg_flag),        
            
            if (BN_flag == 0),
                
                temp_in1to4 = [];   
                temp_in1to4(:,1) = (in1to4(:,3) + in1to4(:,1))./2;
                temp_in1to4(:,2) = (in1to4(:,4) + in1to4(:,2))./2;
                temp_in1to4(:,3) = (in1to4(:,3) - in1to4(:,1));
                temp_in1to4(:,4) = (in1to4(:,4) - in1to4(:,2));
                
                in1to4 = []; 
                in1to4 = temp_in1to4;
                
                model = ['pcj'; 'vcj'; 'pvg'; 'vvg'; ' fr'; ];
                
                disp(' ')
                disp('***************************************')
                disp('COMPUTED MODEL BASED ON CONJ. AND VERG.')
                disp('***************************************')
                disp(' ')
                
            else
                
                temp_in1to4 = [];   
                temp_in1to4(:,1) = (in1to4(:,3) + in1to4(:,1))./2;
                temp_in1to4(:,2) = 0 .* in1to4(:,1);
                temp_in1to4(:,3) = (in1to4(:,3) - in1to4(:,1));
                temp_in1to4(:,4) = 0 .* in1to4(:,1);;
                
                in1to4 = []; 
                in1to4 = temp_in1to4;
                
                model = ['vcj'; '---'; 'vvg'; '---'; ' fr'; ];
                
                disp(' ')
                disp('***************************************')
                disp('COMPUTED MODEL BASED ON CONJ. AND VERG.')
                disp('***************************************')
                disp(' ')
                
            end
            
            
        end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %IF appropriate, build proper reduced model
    
    if (exist('red_model') == 1), %For 2nd-3rd-... runs
        
        for k = 1:4,
            if (red_model(k+1) == 0),
                in1to4(:,k) = in1to4(:,k).*red_model(k+1);
                model(k,:) = '---';
            end
        end
        
        disp(' ')
        disp('************************')
        disp('ORGANISED REDUCED MODEL ')
        disp('************************')
        disp(' ')
        
    end
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %IF required, try to optimize for speed
    
    if(optimize_flag),
        
        disp(' ')
        disp('*****************************************')
        disp('Optimize initial conditions for speed ...')
        disp('*****************************************')
        disp(' ')
        
        pause(1)
        
        j=1; ini = [];
        for k = -6:1:6,
            ini(j,:) = [100 k k k k];
            j = j+1;
        end
        
        outini = []; tout = [];
        for w = 1:size(ini,1),
            mode_flag = [8888 ini(w,1:5)];
            
            tic
            [x,se,error_sim,VAF,BIC,est,Others] = dodyn3(M,ns,lat,out1,in1to4,mode_flag);       
            tout = toc;
            outini(w,:) = [tout ini(w,1:5) VAF x];		%write the VAF in the second column with corresponding latency latency
        end;
        
        best = [];
        best = min(find(outini(:,1) == min(outini(:,1))));
        mode_flag = [8888 outini(best,2:6)];
        
        disp(['Best iteration time (sec): ', num2str(outini(best,1))])
        disp(['Optimal initial conditions: ', num2str(outini(best,2:6))])
        disp(' ')
        
    end
    
    
    %Run bootstrap
    
    if (exist('mode_flag') == 0),
        mode_flag = 0;
    end
    
    f = filename_cont;    
    
    
    [filename_m] = bootstrap(M,ns,lat,out1,in1to4,mode_flag,userdir,model,1,f(1:length(f)-4));
    
    disp(' ')
    disp(['Unit ' , filename_cont , ' done!'])
    disp(' ')
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %IF required, email me 
    
    if(email_flag)
        x = clock;
        subject = ['Unit ' , filename_m(1:length(filename_m)-4) , ' completed;  ' date, ', '  num2str(x(4)) , ':' , num2str(x(5)) ];
        sendmail(current_computer, email_address, subject, 'Automatic message from Matlab!');
    end
    
    
    load('multi_data')
    n_files = size(dir_info,1);
    
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

%MAIN multi-unit loop

if (exist('multi_data.mat','file') == 2),
    
    load('multi_data')
    n_files = size(dir_info,1);
    flag_cont = 0;
    
    clc
    disp('File BS_MULTI loaded. ')
    disp(' ')
    disp('NOTE: all files in the current directory that have the *.MAT extension')
    disp('      will be considered as properly exported files...')
    disp(' ')
    
    disp(' ')
    disp('List of files:')
    disp(' ')
    for k = 1:n_files,
        disp(dir_info(k).name)
    end
    disp(' ')
    
else
    
    dir_info = dir;
    dir_info(1:2) = [];
    for k = 1:size(dir_info,1),
        fname(k,1:3) = dir_info(k).name(1:3);
    end
    
    dir_info(find(fname(:,1) == 'b' & fname(:,2) == 's' & fname(:,3) == '_')) = [];
    n_files = size(dir_info,1);
    for k = 1:n_files,
        dir_info(k).picked = 0;
    end
    
    flag_cont = 0;
    
    save(['multi_data.mat'],'dir_info')
    
    clc
    disp('File BS_MULTI created. ')
    disp(' ')
    disp('NOTE: all files in the current directory that have the *.MAT extension')
    disp('      will be considered as properly exported files...')
    disp(' ')
    
    disp(' ')
    disp('List of files:')
    disp(' ')
    for k = 1:n_files,
        disp(dir_info(k).name)
    end
    disp(' ')
    
end  



i = 1;
while (i <= n_files)
    
    clear dir_info
    load('multi_data')
    
    if (dir_info(i).picked == 1) | (dir_info(i).isdir == 1), %Check if file already picked
        
        i = i+1;
        
    else
        
        disp(['Current file: ' dir_info(i).name])
        disp(' ')
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %NOW, load up the cell 
        
        clear M ns lat out1 in1to4 userdir model mode_flag
        load(dir_info(i).name)
        
        
        %Tell the other computers that this file was taken
        dir_info(i).picked = 1;
        save(['multi_data.mat'],'dir_info')
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %IF required, build conjugate and vergence model
        
        if (verg_flag),        
            
            if (BN_flag == 0),
                
                temp_in1to4 = [];   
                temp_in1to4(:,1) = (in1to4(:,3) + in1to4(:,1))./2;
                temp_in1to4(:,2) = (in1to4(:,4) + in1to4(:,2))./2;
                temp_in1to4(:,3) = (in1to4(:,3) - in1to4(:,1));
                temp_in1to4(:,4) = (in1to4(:,4) - in1to4(:,2));
                
                in1to4 = []; 
                in1to4 = temp_in1to4;
                
                model = ['pcj'; 'vcj'; 'pvg'; 'vvg'; ' fr'; ];
                
                disp(' ')
                disp('***************************************')
                disp('COMPUTED MODEL BASED ON CONJ. AND VERG.')
                disp('***************************************')
                disp(' ')
                
            else
                
                temp_in1to4 = [];   
                temp_in1to4(:,1) = (in1to4(:,3) + in1to4(:,1))./2;
                temp_in1to4(:,2) = 0 .* in1to4(:,1);
                temp_in1to4(:,3) = (in1to4(:,3) - in1to4(:,1));
                temp_in1to4(:,4) = 0 .* in1to4(:,1);;
                
                in1to4 = []; 
                in1to4 = temp_in1to4;
                
                model = ['vcj'; '---'; 'vvg'; '---'; ' fr'; ];
                
                disp(' ')
                disp('***************************************')
                disp('COMPUTED MODEL BASED ON CONJ. AND VERG.')
                disp('***************************************')
                disp(' ')
                
            end
            
            
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %IF appropriate, build proper reduced model
        
        if (exist('red_model') == 1), %For 2nd-3rd-... runs
            
            for k = 1:4,
                if (red_model(k+1) == 0),
                    in1to4(:,k) = in1to4(:,k).*red_model(k+1);
                    model(k,:) = '---';
                end
            end
            
            disp(' ')
            disp('************************')
            disp('ORGANISED REDUCED MODEL ')
            disp('************************')
            disp(' ')
            
        end
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %IF required, try to optimize for speed
        
        if(optimize_flag),
            
            disp(' ')
            disp('*****************************************')
            disp('Optimize initial conditions for speed ...')
            disp('*****************************************')
            disp(' ')
            
            pause(1)
            
            j=1; ini = [];
            for k = -6:1:6,
                ini(j,:) = [100 k k k k];
                j = j+1;
            end
            
            outini = []; tout = [];
            for w = 1:size(ini,1),
                mode_flag = [8888 ini(w,1:5)];
                
                tic
                [x,se,error_sim,VAF,BIC,est,Others] = dodyn3(M,ns,lat,out1,in1to4,mode_flag);       
                tout = toc;
                outini(w,:) = [tout ini(w,1:5) VAF x];		%write the VAF in the second column with corresponding latency latency
            end;
            
            
            best = [];
            best = min(find(outini(:,1) == min(outini(:,1))));
            mode_flag = [8888 outini(best,2:6)];
            
            
            inicond_p = figure('Name','Test for optimal initial conditions');
            subplot(2,1,1)
            plot(outini(:,3),outini(:,1),'.')
            xlabel('Initial Value'); ylabel('Iteration time')
            subplot(2,1,2)
            plot(outini(:,3),outini(:,7),'.')
            xlabel('Initial Value'); ylabel('VAF')
            
            f = dir_info(i).name;
            filename_inicond = ['bs_' f(1:end-4) '_1_inicond.fig'];
            saveas(inicond_p,filename_inicond)
            
            disp(['Best iteration time (sec): ', num2str(outini(best,1))])
            disp(['Optimal initial conditions: ', num2str(outini(best,2:6))])
            disp(' ')
            
            close(inicond_p)
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %NOW, start the bootstrap
        
        
        if (exist('mode_flag') == 0),
            mode_flag = 0;
        end
        
        
        f = dir_info(i).name;
        
        [filename_m] = bootstrap(M,ns,lat,out1,in1to4,mode_flag,userdir,model,0,f(1:length(f)-4));
        
        
        disp(' ')
        disp(['Unit ' , dir_info(i).name , ' done!'])
        disp(' ')
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        %IF required, email me 
        
        if(email_flag)
            x = clock;
            subject = ['Unit ' , filename_m(1:length(filename_m)-4) , ' completed;  ' date, ', '  num2str(x(4)) , ':' , num2str(x(5)) ];
            sendmail(current_computer, email_address, subject, 'Automatic message from Matlab!');
        end
        
        i = i+1;
        
    end
    
end %OF while loop



disp(' ')
disp('___________________________________')
disp('MULTI_UNIT LOOP DONE ...')
disp(' ')
disp(['Total elapsed time (sec): ', num2str(etime(clock,t0))])
disp(' ')





