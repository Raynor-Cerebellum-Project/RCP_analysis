%Loadinic
%
%This simply calculate (or transform) some additionnal
%signals from the raw data.
%
% PAS, August 1997
%

% Check if the loaded file is a new one (from svcalib) that contains the vergence stuff
% If it doesn't exist, chande the To be saved stuff

%variablestobesaved = 'fr ghpos gvpos htar vtar hhpos hvpos vf tach filetype M ns index Ts lat variablestobesaved';

flag_right = 0;

%check for target_offset
disp(' ')
if (filetype(1) == 's'),
   disp('Target was not offset for this stich file');
elseif (exist('target_offset') == 1),
   htar=(htar + target_offset);
   disp(['Target was offset by ',num2str(target_offset)])
else
   disp('Target was NOT offset. Be careful...')
end;
disp(' ')

%check for vergence position correction
if (exist('pverg') == 1),
   path = []; path = [cd '\'];
   fileinfo = dir([path dataname]);
   filedate = fileinfo(1,1).date(:,:);
   filedate = datenum(filedate(1:11));
   
   refdate = datenum('18-Aug-1999'); %date when the Rex2mat algorithm was corrected
   
   if (filedate < refdate),
      pverg = -pverg;
      disp(' ')
      disp(['The file was last modified on ', datestr(filedate)])
      disp('The VERGENCE position signal has been sign corrected!!!')
      disp(' ')
   end;
end;

%check for ua saved in a stich file      %modified oct 7, 1999
if (filetype(1) == 's') & (exist('ua') == 1),
   disp('UA already saved in this stich file!')
else
   ua = ix2bc(index,10,length(ghpos));
end;

%compute velocities
hhv      = diff(hhpos)/Ts'; 
hvv      = diff(hvpos)/Ts'; 
ghv      = diff(ghpos)/Ts'; 
gvv      = diff(gvpos)/Ts';  

%Plotlist = str2mat(' fr' , 'ehp' , 'hhp' , 'ghp' );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OPTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global inioption
inioption = [];

inich;
%disp('Please hit a key...')


if (inioption == 1),       %if classic choice is chosen
   
   global flag_hf ns_control ns_bump
   ns_control = []; ns_bump = [];
   
   disp('Regular Format')
   %pconj = zeros(size(ghpos));		%Create dummy signals.
   %vconj = zeros(size(ghv));		%These signals can be used only if recording the 2 eye signals
   pverg = zeros(size(ghpos));
   vverg = zeros(size(ghv)); 
   pconj = ghpos;
   vconj = ghv;
   ehpos = ghpos - hhpos;
   ehv = ghv-hhv;  
   
   ns = 1;
   
   if timebase~='stac'
      timebase='stac';
   end;
   %if timebase~='tim-'
   %   timebase='tim-';
   %end;
   
   %Plotlist = str2mat( ' fr' , 'ghv' , 'ghp' );
   
elseif (inioption == 2),   % if vergence choice is chosen
   disp('Vergence Behavior Format')
   
   %variablestobesaved = 'fr ua ghpos gvpos htar vtar hhpos hvpos pconj pverg vf tach filetype M ns index Ts lat variablestobesaved';
   
   vconj = diff(pconj)/Ts;
   vverg = diff(pverg)/Ts; 
   
   %Plotlist = str2mat(  'pcj' , 'vcj' , 'pvg' , 'vvg' );
   
   disp(' ')
   disp('In this format, the signals are:')
   disp('	ghpos = ghp = right eye position')
   disp('	ghvel = ghv = right eye velocity')
   disp('	hhpos = hhp = left eye position')
   disp('	hhvel = hhv = left eye velocity')
   disp(' ')
   
   
elseif (inioption == 3),   % if vergence unit choice is chosen
   disp('Vergence Unit Format')
   
   %variablestobesaved = 'fr ua ghpos gvpos htar vtar hhpos hvpos pconj pverg vf tach filetype M ns index Ts lat variablestobesaved';
   
   vconj = diff(pconj)/Ts;
   vverg = diff(pverg)/Ts; 
   
   ehpos = zeros(size(pconj));
   ehv = zeros(size(vconj));
   
   Plotlist = str2mat(  ' fr' , 'pcj' , 'vcj' );
   
   disp(' ')
   disp('In this format, the signals are:')
   disp('	ghpos = ghp = right eye position')
   disp('	ghvel = ghv = right eye velocity')
   disp('	hhpos = hhp = left eye position')
   disp('	hhvel = hhv = left eye velocity')
   disp(' ')
   
      
elseif (inioption == 4),   % if head free choice is chosen
   disp('Head Fixed Format')
   disp('hx correction')
   hx;
   
   Plotlist = str2mat('hhp' , 'ghp' , 'ehp', 'htr');
   
elseif (inioption == 5),   % if head free choice is chosen
   disp('Sinusoidal Pursuit')
   disp('Regular Format')
   pconj = zeros(size(ghpos));		%Create dummy signals.
   vconj = zeros(size(ghv));		%These signals can be used only if recording the 2 eye signals
   pverg = zeros(size(ghpos));
   vverg = zeros(size(ghv)); 
   
   ns = 1:size(M,1);
   
   if timebase~='stac'
      timebase='stac';
   end;
   
   Plotlist = str2mat(  ' fr' , 'pcj' , 'vcj' );
   
   make_sp;
   
elseif (inioption == 6),   % if head free table choice is chosen
   disp('Head Free Table Format')
   
elseif (inioption == 7),   % if head free with BOTH EYES choice is chosen
   disp('Head Free with Both Eyes Format')
   
   variablestobesaved = 'fr ua ghpos gvpos htar vtar hhpos hvpos pconj pverg vf tach filetype M ns index Ts lat variablestobesaved';
   
   fr = htar(1+lat:length(htar)-1); %set fr = htar so we still have an idea of the target position
   headvel = diff(vtar)/Ts;
   htar = smooth(headvel,75,51);
   
   r_temp = gcorr_r(ghpos,vtar);    %correction algorithm for right eye
   ghpos = r_temp;
   ghv = diff(ghpos)/Ts;
   %ghv = smooth(r_vel,75,51);
   
   l_temp = gcorr_l(hhpos,vtar);    %correction algorithm for left eye
   hhpos = l_temp;
   hhv = diff(hhpos)/Ts;
   %hhv = smooth(l_vel,75,51);
   
   gvpos = ghpos - vtar;            % compute eye-in-head positions and velocities
   hvpos = hhpos - vtar;
   gvv = ghv - htar;
   hvv = hhv - htar;
   
   pconj = (ghpos + hhpos)./2;      % re-calculate conj. and verg. using corrected 
   pverg = (gvpos - hvpos);         % eye signals
   
   vconj = diff(pconj)/Ts;
   vverg = diff(pverg)/Ts; 
   %vconj = smooth(sconj,75,51);
   %vverg = smooth(sverg,75,51);	
   
   clear headvel r_temp r_vel l_temp l_vel 
   
   Plotlist = str2mat(  'vcj' , 'vvg' , 'ghv' , 'hhv' , 'htr' );
   legend('Conj. Vel.','Verg. Vel.','Right Gaze','Left Gaze','Head Vel.',-1);
   
   disp(' ')
   disp('In this format, the signals are:')
   disp(' ')
   disp('	ghpos = ghp = right eye gaze position')
   disp('	ghv   = ghv = right eye gaze velocity')
   disp('	hhpos = hhp = left eye gaze position')
   disp('	hhv   = hhv = left eye gaze velocity')
   disp(' ')
   disp(' 	gvpos = gvp = right eye position')
   disp(' 	gvv   = gvv = right eye velocity')
   disp(' 	hvpos = hvp = left eye position')
   disp(' 	hvv   = hvv = left eye velocity')
   disp(' ')
   disp('	vtar  = vtr = horizontal head position (pot)')
   disp('	htar  = htr = horizontal head velocity (pot)')
   disp(' ')
   disp(' 	fr = horizontal target position')
   disp(' ')
   disp('Pretty confusing, Huh!!!')
   
elseif(inioption == 8),      %Cowboy MN choice
   disp('Cowboy MN (LEFT) format')
   
   %variablestobesaved = 'fr ghpos gvpos htar vtar hhpos hvpos pconj pverg vf tach filetype M ns index Ts lat variablestobesaved';
   
   pconj = ghpos;	    %to simplify the analysis, i.e. it fits with the code
   vconj = ghv;	
   pverg = zeros(size(ghpos));     %dummy vergence since not in use in MN analysis code
   vverg = zeros(size(ghv)); 
   
   Plotlist = str2mat(  ' fr' , 'pcj' , 'vcj' );
   
   disp(' ')
   disp('In this format, the signals are:')
   disp('   ghpos = ghp = pconj = pcj =    right eye gaze position')
   disp('	ghvel = ghv = vconj = vcj =    right eye velocity')
   disp('	hhpos = hhp = head position')
   disp('	hhvel = hhv = head velocity')
   disp(' ')
   
elseif(inioption == 9),      %Cowboy MN choice
   disp('Cowboy MN (RIGHT) format')
   
   %variablestobesaved = 'fr ua ghpos gvpos htar vtar hhpos hvpos pconj pverg vf tach filetype M ns index Ts lat variablestobesaved';
   
   pconj = -ghpos;	    %to simplify the analysis, i.e. it fits with the code
   vconj = -ghv;	
   pverg = zeros(size(ghpos));     %dummy vergence since not in use in MN analysis code
   vverg = zeros(size(ghv)); 
   
   Plotlist = str2mat(  ' fr' , 'pcj' , 'vcj' );
   
   disp(' ')
   disp('In this format, the signals are:')
   disp('	-ghpos = -ghp = pconj = pcj =    right eye gaze position')
   disp('	-ghvel = -ghv = vconj = vcj =    right eye velocity')
   disp('	hhpos = hhp = head position')
   disp('	hhvel = hhv = head velocity')
   disp(' ')
   disp('The eye signals were shifted in sign to match the analysis software')
   
elseif(inioption == 10),      %Cowboy MN choice
   disp('Bullfrog MN (RIGHT) format')
   
   %variablestobesaved = 'fr ua ghpos gvpos htar vtar hhpos hvpos pconj pverg vf tach filetype M ns index Ts lat variablestobesaved';
   
   %vconj = diff(pconj)/Ts;
   %vverg = diff(pverg)/Ts; 
   
   pverg = zeros(size(ghpos));
   vverg = zeros(size(ghv)); 
   vconj = ghv;
   pconj = ghpos;
   ehpos = ghpos - hhpos;
   ehv = diff(ehpos).*1000;  

   
   ns = 1:size(M,1);
   
   if timebase~='stac'
      timebase='stac';
   end;
   
   flag_right = 1;
   
   Plotlist = str2mat( ' fr' , 'ghv' , 'ghp' );
   %   Plotlist = str2mat(  ' fr' , 'ghv' , 'hhv' , 'vcj' );
   
end;

clear inioption 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%define the max length of the data:

N = min([length(fr);length(ghv)-lat]);

%Keep track of number of segments

NoofSegments = length(M(:,1));

%Used for sorting
AddFlag = 0;
NewFlag = 0;
