function s = inta2str(n)
%INTA2STR Integer array to string conversion.
%	S = INTA2STR(N)  converts the integer valued vector N
%       into a string representation.
%	See also NUMA2STR, SPRINTF and FPRINTF.
s = [];
if isstr(n)
	s = n;
else
        s = sprintf('%.0f',n(1));
	if length(n)~=1,
	        for k = 2:length(n),
			s = [s, ' ', sprintf('%.0f',n(k))];
	        end
	end
end
