%loaddata - load a new file from a local or network disk
%
%

% 	(c) Claudio G. Rey - 10:13AM  6/30/93


   if exist('dataname') ~=1, dataname = 'none'; end
   [namenew,path] = uigetfile('*.*','Select data file');
   buf001 = [path namenew]; 

   if buf001( 1) ~= 0, 

      dataname = buf001;
%     Load data

      eval(['load ' dataname ' -mat'])

%     Set dummy default Marker if none exists

      if exist('M')~=1, M = [10,100]; end

%     Set default marker pointer if none exists

      if exist( 'ns') ~= 1, ns = 1; end

%     Set default latency if none exists

      if exist( 'lat')~=1, 
         if exist('latency') ~= 1, 
            lat = 20; 
         else 
            lat = latency;
         end,
         lat = 20;
      end

%     Execute initialization module:

      loadini

%     Prepare the plot title:

      NoofDisplayed = length( Plotlist(:,1));

      displaytitle = [dataname ': ']; 
      for i = 1:NoofDisplayed, 
         displaytitle = [ displaytitle Plotlist(i,1:3) ' - '];
      end

%     Prepare a string containing the signal definitions

      plotstr = plotmate( Plotlist, Signals, Signaldefinitions);
 
      replot;

      fileload = 1;

   end
