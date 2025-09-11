
  disp('Calibrating gaze horizontal position')
  ghpos = .69*( array(:,2))/409*20/16 + 5;

  disp('Calibrating gaze vertical position')
  gvpos = .69*( array(:,3))/409*20/16  - .5;

  disp('Calibrating head horizontal position')
  hhpos = (array(:,4) - 192)/409*20/16;

  disp('Calibrating head vertical position')
  hvpos = (array(:,5) - 3200)/409*20/16;

  disp('Calibrating firing rate')
  fr    =  array(:,1)/16*0.475;

  clear array
  calibrate = 'yes',

  disp('Smoothing begins >>>>>>')
  %disp('Non linear smoothing gaze (will take quite a while) ...')
  %ghpos13 = despike3(ghpos',13);
  %ghpos13 = ghpos13';

  B = fir1(51,.125);

  disp('Smoothing gaze ...')
  ghp = filtfilt(B,1,ghpos);
  gvp = filtfilt(B,1,gvpos);

  disp('Smoothing head ...')
  hhp = filtfilt(B,1,hhpos);
  hvp = filtfilt(B,1,hvpos);

  disp('Smoothing ends >>>>>>')
  disp(' ')

  disp('Classification begins >>>>>>')
  disp('Computing flag ...')
  flag1 = .5 - .5*sign(diff(ghp)*1000-20);
  flag2 = .5 + .5*sign(1-abs(diff(ghp))./(3*abs(diff(gvp))+.001));
  disp('Computing markers ...')
  M = fl2mx(flag1+flag2); M = cull(M,25);
  disp('Classification ends <<<<<<')
  disp(' ')

  Ts       = 1/1000;
  ns       = 1;

  clear BB ghposff ghpos gvpos hhpos hvpos flag flag1 flag2

  disp('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
  disp('Compute new firing rate name = rate5')
  disp('This will take a while ......')
  ufd = firing(index,600000,.005,.0001,[-3,3],10);
  disp('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

%  disp('Compute new firing rate name = rate10')
%  disp('This will take a while ...')
%  rate10 = firing(index,600000,.01,.0001,[-3,3],10);
