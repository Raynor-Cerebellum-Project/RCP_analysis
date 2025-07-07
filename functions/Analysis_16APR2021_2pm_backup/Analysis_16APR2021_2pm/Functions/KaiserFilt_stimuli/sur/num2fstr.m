function strout = num2fstr(num,len)

  blank  = '                                          ';
  strin  = num2str(num);
  strlen = length(strin);
 
  strout = [blank(1:floor((len-strlen)/2)) strin blank(1:ceil((len-strlen)/2))];

end