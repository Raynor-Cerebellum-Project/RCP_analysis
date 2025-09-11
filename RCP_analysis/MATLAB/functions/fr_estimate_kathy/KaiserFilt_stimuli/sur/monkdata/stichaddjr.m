%STICHADD - stitch a new data set to a stich file
%
%	This is a low level function that must be edited if the name of the basic variables
%	is changed.
%

%	(c) Claudio G. Rey - 4:46PM  6/23/93

   %if exist( 'stichfilename') ~= 1, stichfilename = '*.sch'; end

   %[namenew,path] = uigetfile( stichfilename, 'Select stich file');
   %stichfilename = [path namenew];

   disp('Wait ...')

   [Md, Mx] = segments(M, ns, 'stac', buffer, pan, N);
   ixd = mx2ix( Md); ixx = mx2ix( Mx);

   indexsave = index; nssave = ns;
   eval(['save ' userdir 'temp Mx Md N ' variablestobesaved]);

   ghpnew = ghpos(ixd);
   gvpnew = gvpos(ixd);
   hhpnew = hhpos(ixd);
   hvpnew = hvpos(ixd);
   htarnew = htar(ixd);
   vtarnew = vtar(ixd);
   tachnew = tach(ixd);
   frnew = fr(ixd);

   Stichfileloaded = 1;
   eval(['load ',stichfilename,' -mat'],'Stichfileloaded = 0;')

   if Stichfileloaded == 1,

 %     disp('This file contains stiched data from the following data files:')

      [numberoffiles,cc] = size( datafilenames);

      for k = 1:numberoffiles, disp(deblank(datafilenames(k,:))); end

      datafilenames = str2mat( datafilenames, dataname);

      variablestobesaved = 'fr ghpos gvpos htar vtar hhpos hvpos vf tach filetype M ns index Ts lat datafilenames variablestobesaved';

 %     disp('Scanning the unit activity ...')

      N   = length(fr);
      for j = 1:length(indexsave),
         ix    = indexsave( j);   
         base  = (N-1)*10;

         for k = 1:length(nssave)
           first = (Md( k, 1)-1)*10;
           last  = (Md( k, 2)-1)*10;

            if ix > first, if ix < last,
               index = [index, (ix - first + base)];
            end, end

            base  = base + (Md( k, 2) - Md( k, 1) + 1)*10;
         end

      end

 %     disp('Stiching the other basic signals ...')

      ghpos = [ghpos; ghpnew];
      gvpos = [gvpos; gvpnew];
      hhpos = [hhpos; hhpnew];
      hvpos = [hvpos; hvpnew];
      htar = [htar; htarnew];
      vtar = [vtar; vtarnew];
      tach = [tach; tachnew];

      fr = [fr; frnew];
      M   = [  M; (Mx+N)];
      N   = length(fr);
      ns  = 1:length(M(:,1));

      filetype = 'stiched';

      %disp(['Segments # ' inta2str(nssave) ' appended to stich file ' stichfilename ' ...'])
      %disp(['saving ' stichfilename ' ...'])
      eval(['save ' stichfilename ' ' variablestobesaved],['disp(''Could not save stich file'')'])

   end

   eval(['load ' userdir 'temp ']);
   clear indexsave ghpnew gvpnew hhpnew hvpnew htarnew vtarnew tachnew frnew nssave Stichfileloaded ixd ixx

   disp(['OK'])

