%STICHADD - stitch a new data set to a stich file
%
%	This is a low level function that must be edited if the name of the basic variables
%	is changed.
%

%	(c) Claudio G. Rey - 4:46PM  6/23/93

SCH_to_save = ['fr ua ghpos gvpos hhpos hvpos pconj pverg vf tach htar vtar yator pitor roltor filetype M ns index Ts lat variablestobesaved;'];


if AddFlag == 0,
    if exist( 'stichfilename') ~= 1, 
        stichfilename = '*.sch'; 
    end
    
    [namenew,path1] = uigetfile( stichfilename, 'Select stich file');
    stichfilename = [path1 namenew];
end;

AddFlag = 0;

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



%Create a backup of workspace and rename raw data

save c:\temp.mat;

M_raw = M; pconj_raw = pconj; pverg_raw = pverg; ghpos_raw = ghpos; gvpos_raw = gvpos; 
hhpos_raw = hhpos; hvpos_raw = hvpos; htar_raw = htar; vtar_raw = vtar; tach_raw = tach;
yator_raw = yator; pitor_raw = pitor; roltor_raw=roltor;
fr_raw = fr; ua_raw = ua;

eval(['load ',stichfilename,' -mat'])


%Set which data points (with buffer) will be saved
cut = []; counter = length(fr); M_new = M; j = size(M,1)+1;
for i = 1:size(kept_ns,2),
    cut = [cut (M_raw(kept_ns(i),1)-SCH_buffer):(M_raw(kept_ns(i),2)+SCH_buffer)];
    M_new(j,1) = counter + SCH_buffer;
    M_new(j,2) = M_new(j,1) + ( M_raw(kept_ns(i),2) - M_raw(kept_ns(i),1));
    counter = counter + (2 * SCH_buffer + M_new(j,2) - M_new(j,1) + 1);
    j = j+1;
end
cut(find(cut(:,:) <= 0)) = 1;
cut(find(cut(:,:) >= length(fr_raw))) = length(fr_raw)-1;

%Now cut the signals
pconj = [pconj; pconj_raw(cut)];
pverg = [pverg; pverg_raw(cut)];
ghpos = [ghpos; ghpos_raw(cut)];
gvpos = [gvpos; gvpos_raw(cut)];
hhpos = [hhpos; hhpos_raw(cut)];
hvpos = [hvpos; hvpos_raw(cut)];
htar  = [htar; htar_raw(cut)];
vtar  = [vtar; vtar_raw(cut)];
yator = [yator; yator_raw(cut)];
pitor = [pitor; pitor_raw(cut)];
roltor = [roltor; roltor_raw(cut)];
tach  = [tach; tach_raw(cut)];
fr    = [fr; fr_raw(cut)];
ua    = [ua; ua_raw(cut)];    
M     = M_new;
ns    = 1:size(M,1);

filetype = 'stiched';


%NOW save it

eval(['save ' stichfilename ' ' SCH_to_save],['disp(''Could not save'')'])


disp('DONE ......')

%Reset initial workspace
clear all 

load c:\temp.mat;
delete c:\temp.mat;

clear textfilename today fid ns1
clear SCH_to_save SCH_buffer kept_ns M_new counter
