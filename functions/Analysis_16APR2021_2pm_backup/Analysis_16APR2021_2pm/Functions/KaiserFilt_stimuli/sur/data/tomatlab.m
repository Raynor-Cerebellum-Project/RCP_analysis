function tomatlab( );
%CALIBRAT - load and filter a neuron data file for a generic file
%

% (c) Claudio G. Rey - 4:15PM  6/25/93
  
   disp(' This routine is to be run only once')
   disp(' After the data is converted all the variables ')
   disp(' are directly available from the file ') 
   disp(' ')

   [namenew,path] = uigetfile('*.*','Pick first data file');
   nameinfirst = [path namenew];
   firstnumber = sscanf(nameinfirst(length(nameinfirst)-2:length(nameinfirst)),'%d');

   [namenew,path] = uigetfile('*.*','Pick last data file');
   nameinlast  = [path namenew];
   lastnumber = sscanf(nameinlast(length(nameinlast)-2:length(nameinlast)),'%d');

   for sessionnumber = firstnumber:lastnumber,   

      if sessionnumber >= 100, 
         sessionname = int2str(sessionnumber); 
      elseif sessionnumber >= 10,
         sessionname = ['0' int2str(sessionnumber)]; 
      else 
         sessionname = ['00' int2str(sessionnumber)]; 
      end

      namein  = [nameinfirst(1:length(nameinfirst)-3) sessionname];

      eval(['!convert ' namein],Error('Could not do the conversion ...')); 

   end

