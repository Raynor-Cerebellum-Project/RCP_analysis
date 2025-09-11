% Direction - determine directional tuning of neuron
%

% (c) Kathleen E. Cullen - 2:42PM  8/9/93

%  load user directory
% load horizontal amplitude as hor
% load vertical amplitude as ver
% load NOS as NOS
% load duration as dur

 amp = sqrt((hor .* hor) + (ver .* ver));
 dir = atan(ver./hor)* 180 / pi;
 n = length(NOS)

 global dtemp NOStemp durtemp x
% durtemp = dur
 dtemp = dir;
 NOStemp = NOS;

% choose saccades 40 -50 deg
 dtemp = dtemp * nan;
 NOStemp = NOStemp * nan;
 durtemp = durtemp * nan;
 j = 0;
  for i= 1:n
    if (amp(i) >= 5 )
      if ( amp(i) <= 110)
         j = j +1
         dtemp(j) = dir(j);
         NOStemp(j) = NOS(j);
         durtemp(j) = 1;
      end
    end
  end

 dtemp = dtemp(1:j);
 NOStemp = NOStemp(1:j);
 durtemp = durtemp(1:j);
 x0 = [20 200 1 20];
 x = leastsq('errfef',x0);
 
disp(' For saccades 2 - 10 degrees, the fit to the equation: ')
disp('  NOStemp = x(1) + (x(2) * exp ((-1 /2) * ( (dtemp - x(3)) / x(4)).^2 )); ')
disp(' The parameters are:' )
disp( x )
disp(' The error is:' )
disp( mean (errfef(x)) )

ang = 1:361; ang = ang - 181; 
fit = x(1) + (x(2) * exp ((-1 /2) * ( (ang - x(3)) / x(4)).^2 ));
plot( ang , fit, '-', dtemp , NOStemp, '.')
[Y, I] = max(fit(1 : 360))
disp('The preferred direction is......' )
disp(ang(I + 0))
disp(x)


% choose saccades 10 -20 deg


% choose saccades 20 -30 deg


% choose saccades 30 -40 deg


% choose saccades 40 -50 deg


% choose saccades 50 -60 deg


% choose saccades 60 -70 deg

  
