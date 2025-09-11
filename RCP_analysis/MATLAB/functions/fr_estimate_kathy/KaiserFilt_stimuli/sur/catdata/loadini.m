%LOADINI - loadata initialization module
%

% 	(c) Claudio G. Rey - 7:18AM  6/20/93


%     redefine for back compatibility:

%    Patch for version 1:

      if exist( 'hhposf') == 1, 
         hhp = hhposf; clear hhposf
         hgp = ghposf; clear ghposf
         vhp = hvposf; clear hvposf
         vgp = gvposf; clear gvposf
         ufd =  rate5;  clear rate5
         clear fr
      end

%     Patch for version 2:

      if exist( 'ghp') == 1, 
         hgp =  ghp; clear ghp
         vhp =  hvp; clear hvp
         vgp =  gvp; clear gvp
         ufd = rate5;  clear rate5
      end

%     Patch for versions 1 and 2:

      if exist( 'calibrate') ==1, filetype = 'calibrated';clear calibrate; end

      ua       = ix2bc(index,10,length(hgp));
      hhv      = diff(hhp)/Ts'; 
      vhv      = diff(vhp)/Ts'; 
      hgv      = diff(hgp)/Ts'; 
      vgv      = diff(vgp)/Ts'; 

%     define the max length of the data:

      N = min([length(ufd);length(hgv)-lat]);


%     optional speedup definitions (require more menory):

%      hep      = ghposf-hhposf;
%      hev      = ghv-hhv;
%      nos      = cumsum(bc);

   NoofSegments       = length(M(:,1));