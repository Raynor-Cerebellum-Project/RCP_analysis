function [fr,fr_estimate] = make_kaiser5(ua,Pass_Band, Stop_Band, Rp,Rs)

win_mode=menu_AK('Windows Mode','kaiser','Causal','Gaussian');
if win_mode==2
% if strcmp(mode,'Kaiser') || strcmp(mode,'kaiser')
%     Rp=0.1;
%     Rs=60;
%     ITL = 20;
%     dev = [10^(Rp/20), 10^(-Rs/20)];
% 
%     Pass_Band=fc;
%     Stop_Band=fc+1;
%     fcut =[Pass_Band Stop_Band];
%     fr=burst(ua,fs,fcut,ITL,dev);
%     fr=fr(1:length(ua));
%     
% elseif strcmp(mode,'Causal') || strcmp(mode,'causal')
fs = 1000;
fc=(Stop_Band+Pass_Band)/2;
    a=fc*9.4352;
    dt=1/fs;
    T=0:dt:2;
    k=a^2*T.*exp(-a.*T);
    k(k<0)=0;
    fr=conv(ua,k);
    fr=fr(1:end-length(T)+1);
    fr_estimate=fr;
elseif win_mode==3
    fs = 1000;
    fc=(Stop_Band+Pass_Band)/2;
%     a=1/(2*pi*fc);
a=0.1325/fc;
    dt=1/fs;
    T=-6*a:dt:6*a;
    k=exp(-(T.^2)/(2*a^2));
    Q = trapz((1:length(k))/1000,k);
%     k(k<0)=0;
    fr=conv(ua,k/Q);
    fr=fr(round(length(T)/2):round(length(T)/2)+ length(ua));
    
    % ################## Check this for dimention mismatch
    fr_estimate=fr(1:end-1);
    fr=fr(1:end-1);

elseif win_mode==1    

fs = 1000;

fcut =[Pass_Band Stop_Band];

ITL = 20;

dev = [10^(Rp/20), 10^(-Rs/20)];

fr_estimate = burst(ua,fs,fcut,ITL,dev);

fr = fr_estimate;
end