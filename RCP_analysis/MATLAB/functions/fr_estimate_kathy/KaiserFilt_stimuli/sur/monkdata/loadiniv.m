%Loadiniv
%
%This simply calculate (or transform) some additionnal
%signals from the raw data.
%
% PAS, August 1997
%
%Modification: Now needs to specify the amount by which the target
%must be offset. Define "target_offset" in the workspace
%This is good for ramps
%
% PAS, July 1998


% Check if the loaded file is a new one (from svcalib) that contains the vergence stuff
% If it doesn't exist, chande the To be saved stuff

if (exist('target_offset') == 0),
   disp(' ')
   disp('You did not specify the offset of the target')
   disp('********************************************')
   disp(' ')
   return;
end;

if (filetype(1) == 's') & (exist('ua') == 1),       %modified oct 7, 1999
else
   ua = ix2bc(index,10,length(ghpos));
end;

htar = htar + target_offset;

if exist('pconj') == 0	%If we are dealing with an old file which doesn't contain all the stuff calculated in svcalnew
   
   hhv      = diff(hhpos)/Ts'; 
   hvv      = diff(hvpos)/Ts'; 
   ghv      = diff(ghpos)/Ts'; 
   gvv      = diff(gvpos)/Ts';      
   
   
   pconj = ghpos;  
   vconj = ghv;
   pverg = zeros(size(ghpos));
   vverg = zeros(size(ghv)); 
   
   disp('Load Hf file')
   
   Plotlist = str2mat('hhv' , 'hvv' , 'ghv' , 'gvv' );
   
else	%If dealing with a file that contains all the variables in svcalnew
   
   %Vergence correction if necessary
   path = []; path = [cd '\'];
   fileinfo = dir([path dataname]);
   filedate = fileinfo(1,1).date(:,:);
   filedate = datenum(filedate(1:11));

   refdate = datenum('18-Aug-1999'); %date when the Rex2mat algorithm was corrected
   
   if (filedate < refdate),
      pverg = -pverg;
      disp('The VERGENCE position signal has been sign corrected!!!')
   end;
   
   hhv      = diff(hhpos)/Ts'; 
   hvv      = diff(hvpos)/Ts'; 
   ghv      = diff(ghpos)/Ts'; 
   gvv      = diff(gvpos)/Ts'; 
   
   vconj = diff(pconj)/Ts;
   vverg = diff(pverg)/Ts; 
   
   Plotlist = str2mat( 'pcj' , 'vcj' , 'pvg' , 'vvg' );
   
end;



%define the max length of the data:

N = min([length(fr);length(ghv)-lat]);


%Keep track of number of segments

NoofSegments = length(M(:,1));

%Used for sorting
AddFlag = 0;
NewFlag = 0;

buffer = 300;