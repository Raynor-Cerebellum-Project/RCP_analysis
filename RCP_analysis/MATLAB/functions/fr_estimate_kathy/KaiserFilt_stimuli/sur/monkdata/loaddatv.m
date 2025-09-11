%loaddata - load a new file from a local or network disk
%
%

% 	(c) Claudio G. Rey - 10:13AM  6/30/93


if exist('dataname') ~=1, dataname = 'none'; end
[namenew,path] = uigetfile('*.*','Select data file');
buf001 = [namenew]; 

if (namenew == 0),
   disp('User aborted...')
   abort_flag = 1;
   clear hvpod htar hhpos gvpos ghv ghpos fr filetype dynaparfilename dataname data colorlist buffer buf001 basicdefaultsignal ans Ts Signals
   clear Signaldescriptions Signaldefinitions vverg vtar vf vconj variablestobesaved vanparfilename userdir timebase threshparfilename plotfilaname
   clear tach pverg polyparfilename plotfilename pconj path pan ns
   clear M hvpos mbiasparfilename NoofSignals index namenew Plotlist lat
else
   if buf001( 1) ~= 0, 
      
      dataname = buf001;
      %     Load data
      
      eval(['load ' dataname ' -mat'])
      
      %     Set dummy default Marker if none exists
      
      if exist('M')~=1, M = [10,100]; end
      
      %     Set default marker pointer if none exists
      
      if exist( 'ns') ~= 1, ns = 1; end
      
      %     Set default latency if none exists
      
      if (exist('lat') ~= 1) & (exist('latency') ~= 1), 
         lat = 0; 
      elseif (exist('latency') == 1), 
         lat = latency;
      end;
      
      %     Execute initialization module:
      
      %loadleo;
      %pconj = zeros(size(ghpos));
      %vconj = zeros(size(ghv));
        
      loadinic;   
          
      %  loadiniv;
      
      %     Prepare the plot title:
      
      NoofDisplayed = length( Plotlist(:,1));
      
      displaytitle = [dataname ': ']; 
      for i = 1:NoofDisplayed, 
         displaytitle = [ displaytitle Plotlist(i,1:3) ' - '];
      end
      
      %     Prepare a string containing the signal definitions
      
      plotstr = plotmate( Plotlist, Signals, Signaldefinitions);
      
      if (exist('hdatadisplay') == 0),
         hdatadisplay = figure('Name', 'Data Display', 'Position', [65   224   896   503],'NumberTitle','off','Color',[0,0,.05],'toolbar','none');
      end;
      
      replot;
      
      fileload = 1;
      
   end
end
