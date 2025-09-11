


ounit = fr([M(1,1)]:[M(1,2)]);
x = length(M);

for i=2:(x-1)

len1 = length(ounit);
len2 = length(fr([M((i+1),1)]:[M((i+1),2)]));
len3 = len1 + len2;

unit = 1:len3;
unit = unit';

unit(1:len1) = ounit;
unit((len1+1):len3) = fr([M((i+1),1)]:[M((i+1),2)]);
ounit = unit;

end
plot(unit)
std(unit)
mean(unit)

