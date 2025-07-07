function [Signaldefinitions, Signals, Signaldescriptions] = listiniv();
%
%
%	
%
%

% 	Claudio G. Rey - 9:21AM  7/28/93


%  Signal definitions


   Signals = ' fr'; 
   Signaldescriptions = 'unit firing density';
   Signaldefinitions  = 'fr';
   
   Signals = str2mat(Signals,' ua');
   Signaldescriptions = str2mat( Signaldescriptions, 'unit activity');
   Signaldefinitions  = str2mat( Signaldefinitions, 'ua');

   Signals(  3, 1:3) = 'nos';
   Signaldescriptions = str2mat( Signaldescriptions, 'cum. no. of spikes');
   Signaldefinitions  = str2mat( Signaldefinitions, 'cumsum(ua)');

   Signals(  4, 1:3) = 'hvv';
   Signaldescriptions = str2mat( Signaldescriptions, 'ver head vel');
   Signaldefinitions  = str2mat( Signaldefinitions, 'hvv(1+lat:N)');

   Signals(  5, 1:3) = 'gvv';
   Signaldescriptions = str2mat( Signaldescriptions, 'ver gaze vel');
   Signaldefinitions  = str2mat( Signaldefinitions, 'gvv(1+lat:N)');

   Signals(  6, 1:3) = 'ehv';
   Signaldescriptions = str2mat( Signaldescriptions, 'hor eye vel');
   Signaldefinitions  = str2mat( Signaldefinitions, 'ghv(1+lat:N)-hhv(1+lat:N)');
 
   Signals(  7, 1:3) = 'hhv';
   Signaldescriptions = str2mat( Signaldescriptions, 'hor head vel');
   Signaldefinitions  = str2mat( Signaldefinitions, 'hhv(1+lat:N)');

   Signals(  8, 1:3) = 'ghv';
   Signaldescriptions = str2mat( Signaldescriptions, 'hor gaze vel');
   Signaldefinitions  = str2mat( Signaldefinitions, 'ghv(1+lat:N)');

   Signals(  9, 1:3) = 'hvp';
   Signaldescriptions = str2mat( Signaldescriptions, 'ver head pos');
   Signaldefinitions  = str2mat( Signaldefinitions, 'hvpos(1+lat:N)');

   Signals( 10, 1:3) = 'gvp';
   Signaldescriptions = str2mat( Signaldescriptions, 'ver gaze pos');
   Signaldefinitions  = str2mat( Signaldefinitions, 'gvpos(1+lat:N)');

   Signals( 11, 1:3) = 'ehp';
   Signaldescriptions = str2mat( Signaldescriptions, 'hor eye pos');
   Signaldefinitions  = str2mat( Signaldefinitions, 'ghpos(1+lat:N)-hhpos(1+lat:N)');

   Signals( 12, 1:3) = 'hhp';
   Signaldescriptions = str2mat( Signaldescriptions, 'hor head pos');
   Signaldefinitions  = str2mat( Signaldefinitions, 'hhpos(1+lat:N)');

   Signals( 13, 1:3) = 'ghp';
   Signaldescriptions = str2mat( Signaldescriptions, 'hor gaze pos');
   Signaldefinitions  = str2mat( Signaldefinitions, 'ghpos(1+lat:N)');

   Signals( 14, 1:3) = 'htr';
   Signaldescriptions = str2mat( Signaldescriptions, 'hor target pos');
   Signaldefinitions  = str2mat( Signaldefinitions, 'htar(1+lat:N)');

   Signals( 15, 1:3) = 'vtr';
   Signaldescriptions = str2mat( Signaldescriptions, 'ver target pos');
   Signaldefinitions  = str2mat( Signaldefinitions, 'vtar(1+lat:N)');

   Signals( 16, 1:3) = 'pcj';
   Signaldescriptions = str2mat( Signaldescriptions, 'conj pos');
   Signaldefinitions  = str2mat( Signaldefinitions, 'pconj(1+lat:N)');

   Signals( 17, 1:3) = 'vcj';
   Signaldescriptions = str2mat( Signaldescriptions, 'conj vel');
   Signaldefinitions  = str2mat( Signaldefinitions, 'vconj(1+lat:N)');

   Signals( 18, 1:3) = 'pvg';
   Signaldescriptions = str2mat( Signaldescriptions, 'verg pos');
   Signaldefinitions  = str2mat( Signaldefinitions, 'pverg(1+lat:N)');

   Signals( 19, 1:3) = 'vvg';
   Signaldescriptions = str2mat( Signaldescriptions, 'verg vel');
   Signaldefinitions  = str2mat( Signaldefinitions, 'vverg(1+lat:N)');

   Signals( 20, 1:3) = 'trx';
   Signaldescriptions = str2mat( Signaldescriptions, 'xtrans');
   Signaldefinitions  = str2mat( Signaldefinitions, 'xtrans(1+lat:N)');
   
   Signals( 21, 1:3) = 'try';
   Signaldescriptions = str2mat( Signaldescriptions, 'ytrans');
   Signaldefinitions  = str2mat( Signaldefinitions, 'ytrans(1+lat:N)');
   
   Signals( 22, 1:3) = 'trz';
   Signaldescriptions = str2mat( Signaldescriptions, 'ztrans');
   Signaldefinitions  = str2mat( Signaldefinitions, 'ztrans(1+lat:N)');

  NoofSignals = 22;

