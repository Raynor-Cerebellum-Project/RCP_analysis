function newM = chop(M,N1,N2)
% CHOP	Chops off segments before t=N1 and after t=length-N2
%
%	newM = chop(M,N1,N2)
%
%	newM	-	updated marker domain descriptor		
%

% (c) Claudio G. Rey 1991-12-10

  ns = length(M(:,1));

  first = 1;
  for j = 1:ns-1,
     if M(j,1) <N1,
        if M(j+1,1) >N1,
           first = j+1;
        end
     end
  end
  last = ns;
  for j = ns:-1:2,   
     if M(j,2) >N2,
        if M(j-1,2) <N2,
           last = j-1;
        end
     end
  end

  newM = M(first:last,1:2);
end