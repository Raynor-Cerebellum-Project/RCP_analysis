%STICHNEW - start up a new stich file.
%
%	This is a low level function that must be edited if the name of the basic variables
%	is changed.
%

%	(c) Claudio G. Rey - 2:18PM  6/29/93

   %[namenew,path] = uiputfile( '*.sch', 'Select stich file');
   stichfilename = [namenew];

   %disp('Wait ...')

   [Md, Mx] = segments(M, ns, 'stac', buffer, pan, N);
   ixd = mx2ix( Md); ixx = mx2ix( Mx);

   indexsave = index; nssave = ns;
   eval(['save ' userdir 'temp Mx Md N ' variablestobesaved]);

   datafilenames = dataname;

   %variablestobesaved = 'fr ghpos gvpos htar vtar hhpos hvpos vf tach filetype M ns index Ts lat datafilenames variablestobesaved';

   index = [];

   %disp('Scanning the unit activity ...')

   for j = 1:length(indexsave),
      ix    = indexsave( j);   
      base  = 0;

      for k = 1:length(ns)
         first = (Md( k, 1)-1)*10;
         last  = (Md( k, 2)-1)*10;

         if ix > first, if ix < last,
            index = [index, (ix - first + base)];
         end, end

         base  = base + (Md( k, 2) - Md( k, 1) + 1)*10;
      end

   end

   %disp('Adding the other basic signals ...')

   ghpos = ghpos(ixd);
   gvpos = gvpos(ixd);
   hhpos = hhpos(ixd);
   hvpos = hvpos(ixd);
   htar = htar(ixd);
   vtar = vtar(ixd);
   tach = tach(ixd);
   
   fr = fr(ixd);
   M   = Mx;
   ns  = 1:length(ns);

   filetype = 'stiched';

   %disp(['Segments # ' inta2str(nssave) ' appended to stich file ' stichfilename ' ...'])
   %disp(['saving ' stichfilename ' ...'])
   eval(['save ' stichfilename ' ' variablestobesaved],['disp(''Could not save'')'])

   eval(['load ' userdir 'temp ']);
   clear indexsave

   %disp(['OK'])
 