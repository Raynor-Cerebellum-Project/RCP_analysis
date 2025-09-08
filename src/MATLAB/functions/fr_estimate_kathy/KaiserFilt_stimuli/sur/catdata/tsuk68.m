
  disp('gaze horizontal position')
  ghpos = .87*((array(:,2)-79)/409*20*.77/8-11.5)+2.34;

  disp('gaze vertical position')
  gvpos = .87*((array(:,3) - 219)/409*20*.77/8-11.5)+6;

  disp('head horizontal position')
  hhpos = (array(:,4) +  50)/409*20/8 + 2.34;

  disp('head vertical position')
  hvpos = (array(:,5) +  20)/409*20/8 + 6;

  disp('load firing rate')
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

  disp('Computing flag ...')
  flag1 = .5 + .5*sign(diff(ghposf)*1000+20);
  flag2 = .5 + .5*sign(1-abs(diff(ghposf))./(3*abs(diff(gvposf))+.001));
  disp('Computing markers ...')
  M = fl2mx(flag1+flag2); M = cull(M,25);

  disp('Classification ends >>>>>>')
  disp(' ')

  Ts       = 1/1000;
  ns       = 1;
  lat      = 20;

  disp('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
  disp('Compute new firing rate name = rate5')
  disp('This will take a while ......')
  rate5 = firing(index,600000,5/1000,1/10000,[-3,3],10);
  disp('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

%  disp('Compute new firing rate name = rate10')
%  disp('This will take a while ...')
%  rate10 = firing(index,600000,.01,.0001,[-3,3],10);
