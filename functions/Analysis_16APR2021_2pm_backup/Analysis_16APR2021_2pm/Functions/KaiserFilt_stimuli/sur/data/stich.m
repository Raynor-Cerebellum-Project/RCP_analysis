function stich( stchfile, datafile, data, Ts);
%STICH - stitch a new data set to a stich file
%
%	stich( stchfile, datafile, data, Ts);
%
%

%	(c) Claudio G. Rey - 2:40PM  5/11/93

   datall = data;

%disp('%STICH-I-keyboard'),keyboard

   if exist( stchfile)==2,

      eval(['load ',stchfile])
      disp([stchfile,' exists, append to file.'])
 
      nfiles = nfiles + 1;  
      t = ['fileno', int2str(nfiles), ' = ''', datafile, ''';'];
      eval(t); clear t

   else

      disp([stchfile,' created.'])
      nfiles = 1;
      fileno1 = datafile;
      data = [];

   end

   data = [data;datall];

   [rr,cc] = size(data);
   data(:,1) = (1:rr)'*Ts;
   M = fl2mx( sign( data( :, cc)),1);

   clear datall rr cc datafile
   eval(['save ' stchfile])

