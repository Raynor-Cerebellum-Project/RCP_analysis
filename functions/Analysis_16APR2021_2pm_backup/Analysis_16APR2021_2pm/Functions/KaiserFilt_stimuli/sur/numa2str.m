function t = numa2str(x)
%NUMA2STR Number to string conversion.
%	T = NUM2STR(X)  converts the vector  X into a string
%	representation  T  with about  4  digits and an exponent if
%	required.   This is useful for labeling plots with the
%	TITLE, XLABEL, YLABEL, and TEXT commands.  See also INT2STR,
%	SPRINTF, and FPRINTF.

   t = 'none';
   if isstr(x),
      t = x;
   else
      if isempty( x)
         t = 'none';
      else
         t = sprintf('%.4g',x(1));
	 if length(x)~=1,
	    for k = 2:length(x),
                t = [t, '  ', sprintf('%.4g',x(k))];
            end
         end
      end
   end
