%ANAINIV - analysis program first initialization module
%

% 	(c) Claudio G. Rey - 11:32AM  1/22/94


%  Default Variables to be save in a data file:

   variablestobesaved = 'fr ua ghpos gvpos htar vtar hhpos hvpos pconj pverg xtrans ytrans ztrans yator pitor roltor vf tach filetype M ns index Ts lat variablestobesaved';

%  Signal to be displayed if the plotdata file is destroyed:

   basicdefaultsignal = 'pcj';

%  Where to store dynamic parameters:

   dynaparfilename    = [userdir 'dynadata.mat'];

%  Where to store polynomial parameters:

   polyparfilename    = [userdir 'polydata.mat'];

%  Where to store multiple bias parameters:

   mbiasparfilename   = [userdir 'mbiasdata.mat'];

%  Where to store polynomial parameters:

   vanparfilename     = [userdir 'vandata.mat'];

%  Where to store thresholding info:

   threshparfilename  = [userdir 'thshdata.mat'];

%  Where to store plotting and signalling choices:

   plotfilename    = [userdir 'plotdata.mat'];
