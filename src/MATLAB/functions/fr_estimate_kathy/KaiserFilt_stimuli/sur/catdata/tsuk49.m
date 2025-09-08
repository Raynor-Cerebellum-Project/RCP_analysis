  
      disp('Calibrating gaze horizontal position')
      ghpos = .69 * array(:,2) / 409 * 20 / 16;

      disp('Calibrating gaze vertical position')
      gvpos = .69 * array(:,3) / 409 * 20 / 16 - 19;

      disp('Calibrating head horizontal position')
      hhpos = array(:,4) / 409 * 20 / 16 - 2.3;

      disp('Calibrating head vertical position')
      hvpos = array(:,5) / 409 * 20 / 16 + 4.5;

%     disp('Calibrating firing rate')
%     fr    =  array(:,1)/16*0.475;

      clear array
      filetype = 'calibrated',

%     200 Hz filter

      B = fir1( 51, 0.125);

      disp('Smoothing gaze ...')
      hgp = filtfilt(B,1,ghpos);
      vgp = filtfilt(B,1,gvpos);

      disp('Smoothing head ...')
      hhp = filtfilt(B,1,hhpos);
      vhp = filtfilt(B,1,hvpos);

      disp(' ')
 
      disp('Classification begins >>>>>>')
      M  = mark( diff(hgp)*1000, 20, 999999,20);
      M = cull(M,25);
      disp(' ')

      Ts   = 1/1000;
      ns   = 1;
      lat  = 20;
      disp( 'Latency defaulted to 20 ms...')

      clear BB ghposff ghpos gvpos hhpos hvpos flag

      disp('Compute new firing rate name = ufd')
      disp('This will take a while ......')
      ufd = firing(index,600000,.005,.0001,[-3,3],10);

%     disp('Compute new firing rate name = rate10')
%     disp('This will take a while ...')
%     rate10 = firing(index,600000,.01,.0001,[-3,3],10);
