function strout = str2fstr(strin,len)

  blank  = '                                          ';
  strlen = length(strin);
 
  strout = [blank(1:floor((len-strlen)/2)) strin blank(1:ceil((len-strlen)/2))];

end