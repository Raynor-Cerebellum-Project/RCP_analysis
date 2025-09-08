function newM = cull(M,N)

% Function cull
%
% This function produces identical results to 'cull' from 'analysis',
% but in about one fifth the time.
%
% Usage:	newM = cull(M,N);
%
% Only segments in M which are longer than N points are kept.
% The new marker list is output as newM.
%

%
% GAW 8/23/97
%

ind = find( (M(:,2)-M(:,1)) > N );
newM = M(ind,:);