function [wi,p]=icw(f,lf,v,M,FT,p)
% ICW	Fit initial value transients.
%
%   [wi,p] = icw(f,lf,v,M,FT)
%
%	FT=1,2 -> initial conditions fitted to data.
%	FT=3,4 -> initial conditions derived from initial output.
%	p = initial conditions.  
%	ic are the initial condition trajectories.
%
%	Fits initial conditions to segmented data.
%

%  Claudio G. Rey,  8:30AM  3/19/93

% Maximum length of any segment:  

  maxlen  = max(M(:,2)-M(:,1));

% Use formulation #2 as default:

  if nargin<5, FT==2; end
  Conts = FT-floor(FT/2)*2;  

% Number of poles:

  nf = length(f)-1;

% Write out an impulse of the maximum length 
% and filter it:

  I(1) = 1;
  I(2:maxlen+lf+2)= zeros(maxlen+lf+1,1);
  Ifil = filter(1,f,I);

% Prepare for filtering:

  ns = length(M(:,1));
  wi = zeros( size( v));

% Now filter each one of the marked segments:

  if isempty(p),
     for j=1:ns,
        on = M( j, 1); off = M( j, 2);
        R              = zeros(nf,nf);
        F              = zeros(nf,1); 
        phi            = zeros(off-on+1,nf);
        for k=1:nf,
           phi(1:off-on+1,k)=Ifil(lf-k+2:off-on+lf-k+2)';
        end
        R             = R + phi'*phi;
        F             = F + phi'*v(on:off);
        p(j,:)  = (R\F)';
        wi(on-lf:off) = filter(p(j,:),f,I(1:off-on+lf+1));
     end
  else
     for j=1:ns,
        on = M( j, 1); off = M( j, 2);
        yf = conv(v(on-lf:on-lf+nf-1),f);
        p(j,1:nf) = yf(1:nf);
        wi(on-lf:off) = filter(p(j,:),f,I(1:off-on+lf+1));
     end       
  end
end
