%savedata - Save all relevant data variables to disk
%
%

% 	(c) Claudio G. Rey - 12:19AM  6/19/93


   disp(['saving ' dataname ' ...'])
   eval(['save ' dataname ' ' variablestobesaved])
   disp(['OK'])
