

% SCH_to_save = ['fr ua ghpos gvpos hhpos hvpos pconj pverg  vf tach htar vtar yator pitor roltor filetype M ns index Ts lat variablestobesaved;'];
SCH_to_save = ['fr ua ghpos gvpos hhpos hvpos pconj pverg vf tach htar vtar xtrans ytrans ztrans filetype M ns index Ts lat variablestobesaved;'];
%First prompt for a decent filename
if (NewFlag == 0),
    [namenew,path] = uiputfile( '*.sch', 'Select stich file');
    stichfilename = [namenew '.sch'];
end

NewFlag = 0;

%Write Text File Info
ns1 = [','];
for i = 1:size(kept_ns,2)
    ns1=[ns1 , inta2str(kept_ns(i)) , ','];
end

textfilename = [stichfilename '.txt'];
today = date;
fid = fopen(textfilename,'at');
fprintf(fid,'\n %s!\t %s!\t %s!\t %s ;',today,dataname,stichfilename,ns1);
fclose(fid);


%Create a backup of workspace

save C:\MATLAB\R2014a\temp.mat;
% save C:\temp.mat;

%Set which data points (with buffer) will be saved
cut = []; counter = 0; M_new = [];
for i = 1:size(kept_ns,2),
    cut = [cut (M(kept_ns(i),1)-SCH_buffer):(M(kept_ns(i),2)+SCH_buffer)];
    M_new(i,1) = counter + SCH_buffer;
    M_new(i,2) = M_new(i,1) + ( M(kept_ns(i),2) - M(kept_ns(i),1));
    counter = counter + (2 * SCH_buffer + M_new(i,2) - M_new(i,1) + 1);
end
cut(find(cut(:,:) <= 0)) = 1;
cut(find(cut(:,:) >= length(fr))) = length(fr)-1;


%Now cut the signals
pconj = pconj(cut);
pverg = pverg(cut);
ghpos = ghpos(cut);
gvpos = gvpos(cut);
hhpos = hhpos(cut);
hvpos = hvpos(cut);
htar  = htar(cut);
vtar  = vtar(cut);
yator = yator(cut);
pitor = pitor(cut);
roltor = roltor(cut);
tach  = tach(cut);
fr    = fr(cut);
ua    = ua(cut);    
M     = M_new;
ns    = 1:size(M,1);

% accelerometer AK
xtrans    = xtrans(cut);
ytrans    = ytrans(cut);
ztrans    = ztrans(cut);

filetype = 'stiched';


%NOW save it

eval(['save ' stichfilename ' ' SCH_to_save],['disp(''Could not save'')'])

disp('Done ...... SAVE M NOW!!!')

%Reset initial workspace

clear all

load C:\MATLAB\R2014a\temp.mat;
delete C:\MATLAB\R2014a\temp.mat;
% load C:\temp.mat;
% delete C:\temp.mat;

clear SCH_to_save SCH_buffer kept_ns M_new counter
clear textfilename today fid ns1
