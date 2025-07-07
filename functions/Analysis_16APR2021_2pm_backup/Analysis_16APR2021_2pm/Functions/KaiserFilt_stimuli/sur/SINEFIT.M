% Fit a simple sine or pseudo-random waveform
%

% (c) Kathleen E. Cullen - 2:42PM  2/18/95


% load signal, once calibrated (gain and offset), as signal

n = length(signal);

% from here
fs = 250; 
fc = 10;  
B = fir1(51,fc/fs*2);
signal = filtfilt(B,1,signal);
% to here (gives filter code)

velocity  = diff(signal)/(1/fs);
time = [1:n];
time = time*(1/fs);
time1=reshape(time,n,1);

tempsig = signal*nan;
t = time1;
n=n-1;

global tempsig velocity t x f1 f2 f3 f4 f5 

j = 0;

  for i= 1:n
    if (abs(velocity(i)) < 40 )
	 j = j + 1;
         tempsig(j) = signal(i);
         t(j) = time1(i);
     end
  end
 

tempsig = tempsig(1:j);
t = t(1:j);

f1 = .2 *2*pi;
f2 = .3 *2*pi;
f3 = .7 *2*pi;
f4 = 1.1 *2*pi;
f5 = 1.9 *2*pi;

x0 = [10 0  .001 1 .001 0 .001 0 .001 0 .001 ];
x = leastsq('err1',x0);
 
fit=(x(1)+(x(2)*(cos(f1*t)))+(x(3)*(sin(f1*t)))+(x(4)*(cos(f2*t)))+(x(5)*(sin(f2*t)))+(x(6)*(cos(f3*t)))+(x(7)*(sin(f3*t)))+(x(8)*(cos(f4*t)))+(x(9)*(sin(f4*t)))+(x(10)*(cos(f5*t)))+(x(11)*(sin(f5*t))));


dc = x(1); 
g1 =  sqrt((x(2) .* x(2)) + (x(3) .* x(3)));
g2 =  sqrt((x(4) .* x(4)) + (x(5) .* x(5)));
g3 =  sqrt((x(6) .* x(6)) + (x(7) .* x(7)));
g4 =  sqrt((x(8) .* x(8)) + (x(9) .* x(9)));
g5 =  sqrt((x(10) .* x(10)) + (x(11) .* x(11)));

ph1 = atan(x(3)./x(2));
ph2 = atan(x(5)./x(4));
ph3 = atan(x(7)./x(6));
ph4 = atan(x(9)./x(8));
ph5 = atan(x(11)./x(10));


plot( t, tempsig,'r',  t , fit, 'y')

disp('parameter array is' )
disp(x)

disp('the gains are' )
disp(g1, g2, g3, g4, g5)

disp('the phases are')
disp(ph1, ph2, ph3, ph4, ph5)

disp('the dc offset is')
disp(dc)



