%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SECTION 1: Extract the info from the text file%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


pathname = []; filename = []; newstichname = []; oldstichname = []; datedata = [];

pathname = input('Enter PATH (e.g. c:\\pa\\data\\):  ','s'); 
filename = input('Enter TEXT FILENAME.TXT (e.g. test.sch.txt):  ','s'); 
newstichname = input('Enter NEW STICH FILENAME.SCH (e.g. test1.sch):  ','s'); 
oldstichname = filename(1:size(filename,2)-4);

A = []; begin = []; commas = [];
[A] = textread(filename,'%c'); 
A = A';

excl = find(A == '!');
semic = find(A == ';');
commas = find(A == ',');

n = size(semic,2);


filedataname = [];
index = 1;
for i = 1:n
   filedataname(i,:) = A(excl(index)+1:excl(index+1)-1);
   index = index + 3;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SECTION 2: Rebuild the SCH file%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numseg = 0;
name = 1;
buffer = 300;
ide = 3;


for i = 1:n,
   %first, load the data file
   name_in = [];
   fileshort = char(filedataname(i,size(filedataname,2)- 11 : size(filedataname,2)));
   name_in = deblank(fileshort);
   
   %Find appropriate ns for this file
   segments = []; 
   d = []; tp = [];
   d = find( (commas > excl(ide)) & (commas < semic(i)) );
   for j = 1:size(d,2)-1,
      tp = A(commas(d(j))+1 : commas(d(j)+1)-1);
      segments(:,j) = str2num(tp);
   end;
   ide = ide + 3;
   
   
   eval(['load ' name_in])
   
   if exist('M')~=1, M = [10,100]; end
   if exist( 'ns') ~= 1, ns = 1; end
   if exist( 'lat')~=1, 
      if exist('latency') ~= 1, 
         lat = 0; 
      else 
         lat = latency;
      end,
      lat = 0;
   end
   target_offset = 0;
   loadiniv;
   path= [];
   ebn_bump
   
   
   
   %here, do the stich file
   bin = segments;
   y = find(bin == 0);
   bin(y) = [];
   
   numseg = numseg + length(bin);
   buffer = 300;
   
   if ~isempty(bin)
      ns = bin;
      namenew = newstichname;
      stichfilename = [path newstichname];
      disp(['Saccades from file ', name_in])
      disp(['	-> Segments # ', inta2str(bin)])
      dataname = [name_in 'b.mat'];
      if exist(stichfilename) == 0,
         NewFlag = 1;
         newstich;
      else
         AddFlag = 1;
         addstich;
      end;
      dataname = [];
   end;  % of if loop
   
   sessionnumber = sessionnumber + 1;
   
end








