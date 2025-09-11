%Takes care of analysis filtering

global F_Data F_Signals 

disp(' ')
disp(['Filter Cutoff: ', num2str(F_Data(1))]) 
disp(['Filter Slope : ', num2str(F_Data(2))])
disp(' ')

if F_Signals(1) == 1,
   fr = smooth(fr,F_Data(1),F_Data(2));
   disp('FR was filtered')
end
if F_Signals(2) == 1,
   if exist('ehv'),
      ehv = smooth(ehv,F_Data(1),F_Data(2));
      disp('EHV was filtered')
   else
      disp('NOTE:  EHV is not defined!')
   end
end
if F_Signals(3) == 1,
   hhv = smooth(hhv,F_Data(1),F_Data(2));
   disp('HHV was filtered')
end
if F_Signals(4) == 1,
   ghv = smooth(ghv,F_Data(1),F_Data(2));
   disp('GHV was filtered')
end
if F_Signals(5) == 1,
   if exist('ehpos'),
      ehpos = smooth(ehpos,F_Data(1),F_Data(2));
      disp('EHP was filtered')
   else
      disp('NOTE:  EHPOS is not defined!')
   end
end
if F_Signals(6) == 1,
   hhpos = smooth(hhpos,F_Data(1),F_Data(2));
   disp('HHP was filtered')  
end
if F_Signals(7) == 1,
   ghpos = smooth(ghpos,F_Data(1),F_Data(2));
   disp('GHP was filtered')
end
if F_Signals(8) == 1,
   hvv = smooth(hvv,F_Data(1),F_Data(2));
   disp('HVV was filtered')
end
if F_Signals(9) == 1,
   gvv = smooth(gvv,F_Data(1),F_Data(2));
   disp('GVV was filtered')
end
if F_Signals(10) == 1,
   hvpos = smooth(hvpos,F_Data(1),F_Data(2));
   disp('HVP was filtered')
end
if F_Signals(11) == 1,
   gvpos = smooth(gvpos,F_Data(1),F_Data(2));
   disp('GVP was filtered')
end
if F_Signals(12) == 1,
   htar = smooth(htar,F_Data(1),F_Data(2));
   disp('HTAR was filtered')
end
if F_Signals(13) == 1,
   vtar = smooth(vtar,F_Data(1),F_Data(2));
   disp('VTAR was filtered')
end
if F_Signals(14) == 1,
   vconj = smooth(vconj,F_Data(1),F_Data(2));
   disp('VCJ was filtered')
end
if F_Signals(15) == 1,
   pconj = smooth(pconj,F_Data(1),F_Data(2));
   disp('PCJ was filtered')
end
if F_Signals(16) == 1,
   vverg = smooth(vverg,F_Data(1),F_Data(2));
   disp('VVG was filtered')
end
if F_Signals(17) == 1,
   pverg = smooth(pverg,F_Data(1),F_Data(2));
   disp('PVG was filtered')
end

% accelerometer
if F_Signals(18) == 1,
   xtrans = smooth(xtrans,F_Data(1),F_Data(2));
   disp('TRX was filtered')
end
if F_Signals(19) == 1,
   ytrans = smooth(ytrans,F_Data(1),F_Data(2));
   disp('TRY was filtered')
end
if F_Signals(20) == 1,
   ztrans = smooth(ztrans,F_Data(1),F_Data(2));
   disp('TRZ was filtered')
end

disp(' ')  
disp('Done')

replot

clear global F_Data F_Signals