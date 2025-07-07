%LOADINI - loadata initialization module
%

% 	(c) Claudio G. Rey - 7:18AM  6/20/93

%ghpos = (ghpos * 1.06) + 2;
%hhpos = hhpos + 2.5 ;
%htar = htar + 23.8;
%ehpos=ghpos;

%table = cumsum(tach)/1000;
%table = detrend(table);

%hhpos = hhpos * 0;
%ghpos = ehpos + hhpos;



      ua       = ix2bc(index,10,length(ghpos));

      hhv      = diff(hhpos)/Ts'; 
 %     hhv = tach(1:length(hhv)) + hhv;
      hvv      = diff(hvpos)/Ts'; 
      ghv      = diff(ghpos)/Ts'; 
      gvv      = diff(gvpos)/Ts'; 

global ghpos hhpos ghv hhv  % added to be able to model.m function



%     define the max length of the data:

     N = min([length(fr);length(ghv)-lat]);


%     optional speedup definitions (require more menory):

%      ehp      = ghpos-hhpos;
%      ehv      = ghv-hhv;
%      nos      = cumsum(bc);

   NoofSegments       = length(M(:,1));