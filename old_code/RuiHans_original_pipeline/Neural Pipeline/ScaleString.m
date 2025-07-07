% Take a value, which is a grid spacing, and make a string that shows the
% grid spacing using engineering prefix abbreviations for milli, micro,
% etc.
function outstr = ScaleString(val)

outstr = 'NOTSET';

% List of engineering prefixes for milli, micro, etc.
pfx_str = {'m', char(181), 'n', 'p', 'f', 'a'};

% Use log10() to find position of most significant digit.
%
% MUST BE CAREFUL with floor()!! "val" is computed as the *difference*
% between two numbers (two horizontal grid values), and if those numbers
% are large, then val may not be as exact, so we carefully round it to 3
% significant digits.
val = round(val, 3, 'significant');
L10 = floor(log10(val));
% Divide log val by 3 to index into the prefix strings.
pfxidx = ceil(-L10/3);

if L10 >= 0
    outstr = sprintf('%d', val);
elseif pfxidx > length(pfx_str)
    % Engineering notation, exponent is multiple of 3.
    outstr = sprintf('%de%d', round(val * 10^(pfxidx*3)), -pfxidx*3);
else
    outstr = sprintf('%d%s', round(val * 10^(pfxidx*3)), pfx_str{pfxidx});
end

%fprintf('L10: %d, pfxidx: %d  STRING: "%s"\n', L10, pfxidx, outstr);
end
