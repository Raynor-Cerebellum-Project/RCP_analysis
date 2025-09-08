% Direction - determine directional tuning of neuron
%

% (c) Kathleen E. Cullen - 2:42PM  8/9/93

%  load user directory
% load horizontal amplitude as hor
% load vertical amplitude as ver
% load NOS as NOS
% load duration as dur

 amp = sqrt((hor .* hor) + (ver .* ver));
 dir = atan(ver./hor);
 n = length(NOS)

 global dtemp NOStemp durtemp x
 durtemp = dur
 dtemp = dir;
 NOStemp = NOS;

% choose saccades 20 -30 deg
 dtemp = dtemp * nan;
 NOStemp = NOStemp * nan;
 durtemp = durtemp * nan;
 j = 0;
  for i= 1:n
    if (amp(i) >= 10 )
      if ( amp(i) <= 20)
         j = j +1
         dtemp(j) = dir(j);
         NOStemp(j) = NOS(j);
         durtemp(j) = dur(j);
      end
    end
  end

 dtemp = dtemp(1:j);
 NOStemp = NOStemp(1:j);
 durtemp = durtemp(1:j);
 x0 = [10 10 1 10 1];
 x = leastsq('err',x0);
 
disp(' For saccades 2 - 10 degrees, the fit to the equation: ')
disp(' NOStemp = x(1) + x(2) * (cos(dtemp - x(3))) + x(4)*(cos((2 * dtemp) - x(5))) ')
disp(' The parameters are:' )
disp( x )
% disp(' The error is:' )
% disp( err(x) )

ang = 1:361; ang = ang - 181; 
fit = x(1) + (x(2) * (cos((ang/180*pi) - x(3)))) + (x(4)*(cos((2 * ang/180*pi) - x(5))));
plot( ang , fit)
[Y, I] = max(fit(130 : 230))
disp('The preferred direction is......' )
disp(ang(I + 129))







durtemp(j) = 1;
dtemp = dtemp(1:j);
 NOStemp = NOStemp(1:j);
 durtemp = durtemp(1:j);
 x0 = [10 10 1 10 1];
 x = leastsq('err',x0);


disp(' For saccades 2 - 10 degrees, the fit to the equation: ')
disp(' NOStemp = x(1) + x(2) * (cos(dtemp - x(3))) + x(4)*(cos((2 * dtemp) - x(5))) ')
disp(' The parameters are:' )
disp( x )
% disp(' The error is:' )
% disp( err(x) )

ang = 1:361; ang = ang - 181; 
fit = x(1) + (x(2) * (cos((ang/180*pi) - x(3)))) + (x(4)*(cos((2 * ang/180*pi) - x(5))));
plot( ang , fit)
[Y, I] = max(fit(130 : 230))
disp('The preferred direction is......' )
disp(ang(I + 129))
  
