function [fr] = make_parzen(fr,ua,parzen_width)

%  new_fr = unitf(fr)
%
%  fits a gaussian to the fr
%  called by sloweye.m
% 
%  KEC 


%Get the proper values for the Gaussian
width = 5*parzen_width;
x = -width:1:width;
guass = normpdft(x,0,parzen_width);

%Scale the values appropriately 

sigma = parzen_width * .001;

scale=1/(max(guass'));
tempar = conv(ua, guass');
offset = ((length(guass')-1)/2);
pgain = 1/(sqrt(2*pi)*sigma)*scale;
parzen= pgain * (tempar((offset+1):(length(tempar)-offset)));
fr = parzen;




