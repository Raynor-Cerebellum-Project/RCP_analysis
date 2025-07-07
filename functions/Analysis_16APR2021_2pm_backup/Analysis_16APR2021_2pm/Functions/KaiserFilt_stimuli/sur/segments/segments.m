function [Md, Mx, buffer, pan] = segments( M, ns, timebase, buffer, pan, LEN)
%segments - Compute the marker locations in for display and data;
%
%	[Md, Mx, buffer, pan] = segments( M, ns, timebase, buffer, pan, LEN)
%
%	M 	- Markers for the regions of interest with respect to the time domain.
%	Md	- Markers for the display.
%	Mx	- Markers for the regions of interest with respect to the display.
%	LEN	- Length of the data
%	timebase can take on the following values:
%			'tim-'	- continuous time profile
%			'stac'	- stacked time profile
%			'data'	- synonym for 'stac'
%			'xvsy'	- Md coincides with the markers.
%	buffer	- data shown around the region of interest on each side.
%	pan	- shift display with respect to the area of interest.
%

%  Claudio G. Rey - 7:56PM  6/20/93


   ns = nschck(M,ns);   ni = length(ns);

%  M1 are the chosen segment markers in the time domain:

   M1  = M(ns,:);
   ix1 = mx2ix(M1);

   if M1(1,1)<1, 
      M1(1,1)=1; 
      disp('Irst marker cropped no to exceed the beginning of data')
   end

   if M1(ni,2)>LEN, 
      M1(ni,2)=LEN; 
      disp('Last marker cropped no to exceed the end of data')
   end

%  Modify the pan and buffer so the display does not extend over the data boundary

   buffer = min([floor(  LEN - (M1(ni,2)-M1(1,1)+1) )/2;buffer]);
   pan    = min([LEN-M1(ni,2)-buffer;pan]);
   pan    = max([  1-M1( 1,1)+buffer;pan]);

%  Serve time data index, all data and flag segments:

   if     timebase=='tim-',

%     M3 is the whole data interval with respect to the time domain:
      M3(    1, :) = [(M1(    1, 1) - buffer + pan) (M1(   ni, 2) + buffer + pan)]; 

%     M4 are the segment markers with respect to the whole data interval:
      M4           = M1 + 1 - M3(1,1);

      Mx = M4; Md = M3;


%  Serve time data index, all stacked data and segment flag:

   elseif timebase=='stac',

%     M2 are the stacked data markers with respect to the time domain:
      M2( 1:ni, :) = [(M1( 1:ni, 1) - buffer + pan) (M1( 1:ni, 2) + buffer + pan)]; 

%     M8 are the stacked data markers with respect to the stacked data domain:
      M8( 1, :) = M2( 1, 1:2) - M2( 1, 1) + 1;
      for j = 2:ni, M8( j, 1:2 ) = M8( j-1, 2) + M2( j, 1:2) - M2( j, 1) + 1; end

%     M7 are the segment markers with respect to the stacked data domain:
      M7( 1, :) = M1( 1, 1:2) - M2( 1, 1) + 1;
      for j = 2:ni, M7( j, 1:2 ) = M8( j-1, 2) - M2( j, 1) + 1 + M1( j, 1:2); end

      Mx = M7; Md = M2;


%  Same as 'stac':

   elseif timebase=='data',

%     M2 are the stacked data markers with respect to the time domain:
      M2( 1:ni, :) = [(M1( 1:ni, 1) - buffer + pan) (M1( 1:ni, 2) + buffer + pan)]; 

%     M8 are the stacked data markers with respect to the stacked data domain:
      M8( 1, :) = M2( 1, 1:2) - M2( 1, 1) + 1;
      for j = 2:ni, M8( j, 1:2 ) = M8( j-1, 2) + M2( j, 1:2) - M2( j, 1) + 1; end

%     M7 are the segment markers with respect to the stacked data domain:
      M7( 1, :) = M1( 1, 1:2) - M2( 1, 1) + 1;
      for j = 2:ni, M7( j, 1:2 ) = M8( j-1, 2) - M2( j, 1) + 1 + M1( j, 1:2); end

      Mx = M7; Md = M2;


%  Serve marked data only:

   elseif timebase=='xvsy',

      Mlast = 0;
      for k = 1:length(ns)
         Mx(k,1) = Mlast+1; 
         Mx(k,2) = Mx(k,1) + M1(k,2) - M1(k,1); 
         Mlast   = Mx(k,2);
      end

      Md = M1;

   end


