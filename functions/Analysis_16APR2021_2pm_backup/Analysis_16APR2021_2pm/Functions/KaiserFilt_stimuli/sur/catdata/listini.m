function [Signaldefinitions, Signals, Signaldescriptions] = listini();
%
%
%	[Signaldefinitions, Signals, Signaldescriptions] = listini();
%
%

% 	Claudio G. Rey - 9:21AM  7/28/93


%  Signal definitions

   Signals = 'ufd'; 
   Signaldescriptions = 'unit firing density';
   Signaldefinitions  = 'ufd';
   
   Signals = str2mat(Signals,' ua');
   Signaldescriptions = str2mat( Signaldescriptions, 'unit activity');
   Signaldefinitions  = str2mat( Signaldefinitions, 'ua');

   Signals(  3, 1:3) = 'nos';
   Signaldescriptions = str2mat( Signaldescriptions, 'cum. no. of spikes');
   Signaldefinitions  = str2mat( Signaldefinitions, 'cumsum(ua)');

   Signals(  4, 1:3) = 'vhv';
   Signaldescriptions = str2mat( Signaldescriptions, 'ver head vel');
   Signaldefinitions  = str2mat( Signaldefinitions, 'vhv(1+lat:N)');

   Signals(  5, 1:3) = 'vgv';
   Signaldescriptions = str2mat( Signaldescriptions, 'ver gaze vel');
   Signaldefinitions  = str2mat( Signaldefinitions, 'vgv(1+lat:N)');

   Signals(  6, 1:3) = 'hev';
   Signaldescriptions = str2mat( Signaldescriptions, 'hor eye vel');
   Signaldefinitions  = str2mat( Signaldefinitions, 'hgv(1+lat:N)-hhv(1+lat:N)');
 
   Signals(  7, 1:3) = 'hhv';
   Signaldescriptions = str2mat( Signaldescriptions, 'hor head vel');
   Signaldefinitions  = str2mat( Signaldefinitions, 'hhv(1+lat:N)');

   Signals(  8, 1:3) = 'hgv';
   Signaldescriptions = str2mat( Signaldescriptions, 'hor gaze vel');
   Signaldefinitions  = str2mat( Signaldefinitions, 'hgv(1+lat:N)');

   Signals(  9, 1:3) = 'vhp';
   Signaldescriptions = str2mat( Signaldescriptions, 'ver head pos');
   Signaldefinitions  = str2mat( Signaldefinitions, 'vhp(1+lat:N)');

   Signals( 10, 1:3) = 'vgp';
   Signaldescriptions = str2mat( Signaldescriptions, 'ver gaze pos');
   Signaldefinitions  = str2mat( Signaldefinitions, 'vgp(1+lat:N)');

   Signals( 11, 1:3) = 'hep';
   Signaldescriptions = str2mat( Signaldescriptions, 'hor eye pos');
   Signaldefinitions  = str2mat( Signaldefinitions, 'hgp(1+lat:N)-hhp(1+lat:N)');

   Signals( 12, 1:3) = 'hhp';
   Signaldescriptions = str2mat( Signaldescriptions, 'hor head pos');
   Signaldefinitions  = str2mat( Signaldefinitions, 'hhp(1+lat:N)');

   Signals( 13, 1:3) = 'hgp';
   Signaldescriptions = str2mat( Signaldescriptions, 'hor gaze pos');
   Signaldefinitions  = str2mat( Signaldefinitions, 'hgp(1+lat:N)');

   NoofSignals = 13;

end