filedate = dir('conj.sch');
w = d(1,1).date(:,:);
w = w(1:11);

a=datenum(w);
b = datenum(date);

a < b