%Loadiniv
%
%This simply calculate (or transform) some additionnal
%signals from the raw data.
%
% PAS, August 1997
%

% Check if the loaded file is a new one (from svcalib) that contains the vergence stuff
% If it doesn't exist, chande the To be saved stuff


	variablestobesaved = 'fr ghpos gvpos htar vtar hhpos hvpos pconj pverg vf tach filetype M ns index Ts lat variablestobesaved';

	ua = ix2bc(index,10,length(ghpos));

        hhv      = diff(hhpos)/Ts'; 
        hvv      = diff(hvpos)/Ts'; 
        ghv      = diff(ghpos)/Ts'; 
        gvv      = diff(gvpos)/Ts'; 

        pconj = ghpos;
        vconj = ghv;
        pverg = zeros(size(pconj));
        vverg = zeros(size(vconj));
        
        ehpos = ghpos - hhpos;
        ehv = diff(ehpos).*1000;  

        
        
   	Plotlist = str2mat( 'pcj' , 'vvg' , 'vcj' , 'vvg' );




%define the max length of the data:

     N = min([length(fr);length(ghv)-lat]);


%Keep track of number of segments

     NoofSegments = length(M(:,1));

%Used for sorting
     AddFlag = 0;
     NewFlag = 0;
