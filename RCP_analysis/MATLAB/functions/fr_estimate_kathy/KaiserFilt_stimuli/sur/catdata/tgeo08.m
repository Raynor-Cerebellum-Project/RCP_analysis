  disp('Calibrating gaze horizontal position')
  ghpos = .70*( array(:,2) - 915) / 323 / 1.5;

  disp('Calibrating gaze vertical position')
  gvpos = .70*( array(:,3) - 880) / 406 / 1.5;

  disp('Calibrating head horizontal position')
  hhpos = (array(:,4) - 265) / 327;

  disp('Calibrating head vertical position')
  hvpos = (array(:,5) + 472) / 327;

  disp('Calibrating firing rate')
  fr    =  array(:,1)/16*0.475;

  clear array
  calibrate = 'yes',

  disp('Smoothing begins >>>>>>')
  %disp('Non linear smoothing gaze (will take quite a while) ...')
  %ghpos13 = despike3(ghpos',13);
  %ghpos13 = ghpos13';

  B = fir1(51,.1);

  disp('Smoothing gaze ...')
  ghposf = filtfilt(B,1,ghpos);
  gvposf = filtfilt(B,1,gvpos);

  disp('Smoothing head ...')
  hhposf = filtfilt(B,1,hhpos);
  hvposf = filtfilt(B,1,hvpos);

  disp('Smoothing ends >>>>>>')
  disp(' ')

  disp('Classification begins >>>>>>')
  M  = mark(diff(ghposf)*1000,-9999,-20,20);
  M  = xmark(abs(diff(ghposf))./(3*abs(diff(gvposf))+.001),1,99999,20,0);
  M = cull(M,25);
  disp('Classification ends <<<<<<')
  disp(' ')

  Ts   = 1/1000;
  ns   = 1;
  lat  = 20;

  clear BB ghposff ghpos gvpos hhpos hvpos flag

%  disp('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
%  disp('Compute new firing rate name = rate5')
%  disp('This will take a while ......')
  rate5 = firing(index,600000,.005,.0001,[-3,3],10);
%  disp('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

%  disp('Compute new firing rate name = rate10')
%  disp('This will take a while ...')
%  rate10 = firing(index,600000,.01,.0001,[-3,3],10);

