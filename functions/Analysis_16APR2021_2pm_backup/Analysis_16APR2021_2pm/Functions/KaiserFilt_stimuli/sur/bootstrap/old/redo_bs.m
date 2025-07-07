function [fname_d] = redo_bs()

global abort_flag


%First get directory info



dir_info = dir;
n_files = size(dir_info,1);

for i = 1:n_files,
    fname_d(i,:) = ['----------------------------------------'];
end

flag_cont = 0;

clc
disp('NOTE: all files in the current directory that have the *.MAT extension')
disp('      will be considered as properly exported files...')
disp(' ')


i = 3;
while (i <= n_files)
    
    f = dir_info(i).name;
    
    if prod(f(length(f)-2:length(f)) == 'mat') 
        fname_d(i,1:length(f)) = f;
        
        disp(['Current file: ' f])
        disp(' ')
        
        clear M ns lat out1 in1to4 userdir model
        load(f);
        
        if ~exist('mode_flag'),
            mode_flag = 0;
        end
        
        [x,sd,error_sim,VAF,BIC,est,Others] = dodyn2(M,ns,lat,out1,in1to4,mode_flag);
        ORIG = [x sd VAF BIC error_sim Others.x0 Others.exit];
        alpha = 0.05;
        z = -norminv(alpha/2);  %two sided

        filename_sum = ['rd_' f '.txt'];
        
        fid = fopen(filename_sum,'wt'); 
 
        fprintf(fid,'Filename: %s \n\n\n',filename_sum);
        fprintf(fid,'__________________________________________________________________\n');
        fprintf(fid,'*** RE-CALC. ***: Original Data Set Estimates:  (parameter, s.e., 95%% LB conf., 95%% UB conf.)\n\n');
        fprintf(fid,'\t bias\t %0.4f\t %0.4f\t %0.4f\t %0.4f\n', ORIG(1), ORIG(6) , ORIG(1)-z*ORIG(6) , ORIG(1)+z*ORIG(6));
        for j = 2:5,
            fprintf(fid,'\t %s\t %0.4f\t %0.4f\t %0.4f\t %0.4f\n', model(j-1,:) , ORIG(j), ORIG(j+5) , ORIG(j)-z*ORIG(j+5) , ORIG(j)+z*ORIG(j+5));
        end
        fprintf(fid,'\n');
        fprintf(fid,'\t VAF\t %0.4f\n', ORIG(11));
        fprintf(fid,'\t BIC\t %0.4f\n', ORIG(12));
        fprintf(fid,'\t Error\t %0.4f\n', ORIG(13));
        fprintf(fid,'\n\n');
        
        fprintf(fid,'x0 : \t %0.4f\t %0.4f\t %0.4f\t %0.4f\t %0.4f\n', Others.x0);
        fprintf(fid,'Conv. status: \t\t %0.4f', Others.exit);
        fprintf(fid,'\n\n\n');
        
        fclose(fid); 
        
        disp([filename_sum ' was saved...'])
        disp(' ')
        disp(' ')

    end
    
    i = i+1;
    
end

disp(' ')
disp('___________________________________')
disp('MULTI_UNIT LOOP DONE ...')
disp(' ')




