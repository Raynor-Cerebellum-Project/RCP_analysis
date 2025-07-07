 % cd _ibn.dat/munoz/text
 
 load se16a4522.txt
  disp( 'time')
  time = ( se16a4522(:, 1));

  disp( 'unit 1/0')
  ua = ( se16a4522(:, 2));

  disp( 'horizontal gaze position')
  ghpos =  (1)*( se16a4522(:, 3));
  
  
  disp( 'vertical gaze position')
  gvpos = ( se16a4522(:, 4));

  htar= ghpos*0;
  vtar= ghpos*0;
  hhpos= ghpos*0;
  hvpos= ghpos*0;
  vf= ghpos*0;
  tach= ghpos*0;

 
  clear array

  filetype = 'calibrated',

  disp('Smoothing begins >>>>>>')

  fs = 1000; fc = 125;  B = fir1( 51, fc/fs*2);

  disp('Smoothing gaze ...')
  ghpos = filtfilt(B,1,ghpos);
  gvpos = filtfilt(B,1,gvpos);

  disp('Smoothing head ...')
  hhpos = filtfilt(B,1,hhpos);
  hvpos = filtfilt(B,1,hvpos);

  disp('Smoothing ends >>>>>>')
  disp(' ')

  Ts       = 1/1000;
  ns       = 1;
  lat      = 0;
  buffer    =20;

  disp('Classification begins >>>>>>')

  disp('Computing markers ...')

% Choose right burst (20,9999) left burst would have been (-9999,-20);
% Toss out segments shorter than 25 ms ...
  M  = mark( diff( ghpos)*1000, 20,9999, lat); 
   M = cull( M, 15);

  clear fs fc

  disp('Classification ends >>>>>>')
  disp(' ')

  % disp('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
  % disp('Compute new firing rate name = fr')
  % disp('This will take a while ......')
  % fr = firing(index,600000,5/1000,1/10000,[-3,3],10);
  % disp('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

sigma = 5 * .001
guass = [ 0.0020178419 0.0029303948 0.0045856908 0.0071630436 0.010867024 0.015878843 0.022296187 0.030070119 0.038953066 0.048474431 0.057958394 0.066590443 0.073526926 0.078028575 0.079589151 0.078028597 0.073526986 0.066590607 0.057958882 0.04847575 0.038956456 0.030078396 0.022315411 0.015921373 0.010956668 0.0073431935 0.0049309786 0.0035617785 0.0031194747 ];

 scale=1/(max(guass));
 tempar = conv(ua, guass);
 offset = ((length(guass)-1)/2);
 pgain = 1/(sqrt(2*pi)*sigma)*scale;
 parzen= pgain * (tempar((offset+1):(length(tempar)-offset)));
 fr=parzen;
 
   ghpos = sqrt((ghpos*ghpos) + (gvpos*gvpos))
   ghp = ghpos;
 	hhp = hhpos;
 	gvp = gvpos;
 	hvp = hvpos;

      ehp      = ghp-hhp;

   %   ua       = ix2bc(index,10,length(ghpos));
      hhv      = diff(hhp)/Ts'; 
      hvv      = diff(hvp)/Ts'; 
      ghv      = diff(ghp)/Ts'; 
      gvv      = diff(gvp)/Ts'; 
      
      
%     define the max length of the data:

      N = min([length(fr);length(ghv)-lat]);


%     optional speedup definitions (require more menory):


%      ehv      = ghv-hhv;
%      nos      = cumsum(bc);

   NoofSegments       = length(M(:,1));





      
        
      pconj= ghpos;
      vconj = ghv;
