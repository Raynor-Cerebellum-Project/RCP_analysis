function f = errfef(x)

% NOStemp = [1 2 3 4 5];
% dtemp = [10 12 14 16 17];

% global NOStemp dtemp x
 global NOStemp dtemp durtemp 

f = NOStemp - (x(1) + (x(2) * exp ((-1 /2) * ( (dtemp - x(3)) / x(4)).^2 )));
