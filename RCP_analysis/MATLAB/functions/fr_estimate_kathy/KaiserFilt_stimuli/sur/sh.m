function G = sh( F, M, option, gain, offset)
%SH sample and hold.
%
%	G = sh( F, M, option, gain, offset)
%
%       Hold a value of F during each segment
%	outside of the domain G=F;
%
%	F	-	Input  profile
%	G	-	Output profile
%	M	-	Domain markers
%	option	-	value to be held 
%				-1	first value
%				+1	last value
%				-2	minimum value
%				+2	maximum value
%				 0	mean value
%

% (c) Claudio G. Rey  - 12:29PM  12/22/92

   if nargin<5, offset=0; end
   if nargin<4, gain=1; end

   G = F;
   ns = length(M(:,1));

   for k = 1:ns
       if     option == 0
          value = mean(F(M(k,1):M(k,2)));
       elseif option == -2
          value = min(F(M(k,1):M(k,2)));
       elseif option == 2
          value = max(F(M(k,1):M(k,2)));
       elseif option == -1
          value = F(M(k,1));
       elseif option == 1
          value = F(M(k,2));
       end
       G(M(k,1):M(k,2)) = gain*value*ones((M(k,2)-M(k,1)+1),1)+offset;
   end

end