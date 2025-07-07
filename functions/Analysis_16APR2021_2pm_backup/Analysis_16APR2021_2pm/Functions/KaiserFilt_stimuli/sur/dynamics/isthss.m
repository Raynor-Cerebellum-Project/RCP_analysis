function t=isthss(th)
%ISTHSS	Tests if the model structure is of state-space type
%
%	t = isthss(TH)
%
%	TH: The model structure in the THETA-format (See help theta)
%	t is true if TH is of state-space type, else false

%	L. Ljung 10-2-90
%	Copyright (c) 1990 by the MathWorks, Inc.
%	All Rights Reserved.

t=(th(2,7)>20);
