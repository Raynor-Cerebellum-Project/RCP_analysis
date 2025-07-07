function [data, Md, Mx] = dataseg( M, ns, timebase, buffer, pan, Ts, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16)
%DATASEG
%
%	data = dataseg( M, ns, timebase, buffer, pan, Ts, X1, X2,...)
%

%  Claudio G. Rey - 12:39PM  7/1/93

%  Compute the minimum of all the input vectors:

   t = 'XMAX = min([([])';number = nargin-6;
   for k = 1:number, t = [t,';length(X',int2str(k),')']; end, t = [t,']);'];
   eval(t);

   ni = length(ns);  

   [Md, Mx] = segments( M, ns, timebase, buffer, pan, XMAX);

   ixd = mx2ix( Md); ixx = mx2ix( Mx);

%  Serve time data index, all data:

   if     timebase=='tim-',

      data = zeros(length(ixd),number+1);

%     Time index:

      data(:,1) = [(ixd)*Ts]';

%     Serve data:

      for k = 1:number,
         eval(['data(:,1+' int2str(k) ') = X' int2str(k) '(ixd);']);
      end 

%  Serve stacked data with no time indexes:

   elseif timebase=='data',

      data = zeros(length(ixd),number);

%     Serve data:

      for k = 1:number,
         eval(['data(:,1+' int2str(k) ') = X' int2str(k) '(ixd);']);
      end 


%  Serve time data index, all stacked data:

   elseif timebase=='stac',

      data = zeros(length(ixd),number+1);

%     Time index:
      data(:,1) = [(1:length(ixd))*Ts]';

%     Serve data:

      for k = 1:number,
         eval(['data(:,1+' int2str(k) ') = X' int2str(k) '(ixd);']);
      end 


%  Serve marked data only:

   elseif timebase=='xvsy',

      data = [];
      for k = 1:number,
         eval(['data = [data,X' int2str(k) '(ixd)];']);
      end 

   end



