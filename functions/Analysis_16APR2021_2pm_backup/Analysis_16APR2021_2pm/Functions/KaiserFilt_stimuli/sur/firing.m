function rate = firing(index,ixmax,sig,Ts,range,R);
%firing - compute firing rate
%
% Calling sequence
%	rate = firing(index,ixmax,sig,Ts,range,R);
%

% Claudio G. Rey - 9:14AM  9/25/92

   disp('Creating flag ...')
 
   flag = 1 - ix2fl(index,ixmax);

   disp('Computing Parzen coefficients ...')

   [num,den] = parzenw(sig,Ts,range);
 
   disp('Applying Parzen ...')

   rate = filter(num, 1, flag);

   disp('Decimating data ...')

   w = floor(length(num)/2);

   rate = rate((1+w):R:(length(rate)-w))/sig/(2*pi)^(1/2);
  
