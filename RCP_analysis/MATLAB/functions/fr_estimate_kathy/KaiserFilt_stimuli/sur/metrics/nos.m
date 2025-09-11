function n = nos(spix,M,w)
%NOS - Compute the number of spikes inside a marked domain
%
%	n 	= nos( index, M, w)
%	
%       M 	= marker array
%
%       w 	= bin width
%

% (c) Claudio G. Rey - 9:21AM  8/31/92
   

   ix = mx2ix(M);

   bc = ix2bc(spix,w,ix(length(ix)));

   n = sum(bc(ix));

end
