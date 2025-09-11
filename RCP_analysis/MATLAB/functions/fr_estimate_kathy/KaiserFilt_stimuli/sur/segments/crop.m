function newM = crop(M,nw)
% CROP	Crops segments to eliminate filtering artifacts
%
%	newM = crop(M,nw)
%
%	cuts out nw/2 from the beginning and end of each segment.

% (c) Claudio G. Rey 1991-08-09

  ns    = length(M);
  newns = 0;

  for i = 1:ns
     if (M(i,2) - M(i,1)) > (nw-1), 
        newns = newns+1;
        newM(newns,:)  = [M(i,1)+nw/2,M(i,2)-nw/2];
     end
  end
end