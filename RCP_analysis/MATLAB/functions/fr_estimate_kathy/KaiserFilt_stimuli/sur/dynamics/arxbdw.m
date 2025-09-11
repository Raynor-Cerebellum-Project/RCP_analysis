function [b,f,bias,LAM,BIC,Pcap]=arxbdw(z,index,nn,bias,T);
%arxbdw   Computes LS-estimates of ARX-models
%
%       [b,f,bias] = arxbdw(z,ix,nn,bias,T)
%
%	Returnes the estimated parameters of the ARX model
%	y(t) = B(q) u(t-nk) + (1-F(q)) y(t) + bias + e(t)
%	along with estimated covariances and structure information.
%
%       z : 	the output-input data z=[y u], 
%		with y and u as column vectors.
%	ix : 	marks the data that is actually to be used for the
%		identification
%	bias :	if empty the bias is assumed to be zero.
%	NN: 	NN = [na nb nk], the orders and delays of the above model.
%		For a time series, NN=na only.
%
%	Some parameters associated with the algorithm are accessed by
%
%	[b,a,bias,MSE,BIC,P] = arxbdw(z,ix,nn,bias,T)
%

%  Claudio, G. Rey - 12:12PM  1/8/94

%
%  Ncap = number of data points.
%  nz   = number of inputs + 1 (# of outputs)

   [Ncap,nz]= size(z); nu=nz-1;
   Ncap     = length(index);  
   if nz>Ncap,
      error('Data should be organized in column vectors!'),  
      return,
   end
   if length(nn)~=1+2*(nz-1),  
      disp(' Incorrect number of orders specified!'),
      error('see above'), return,
   end

%
%  Compute markers:

   M = ix2mx( index);

%
%  na = order of regression 
%  nb = order of FIR for each input
%  nk = delay for each input  
%  nt  = number of parameters.

   if nz>1,
      na=nn(1);nb=nn(2:1+nu);nk=nn(2+nu:1+2*nu);  
   else 
      na=nn(1); nb=0; nk=0;
   end  
   nt=na+sum(nb);

%
%  Bias term:

   if isempty( bias) ~= 1,
      nt = nt + 1;
   end

%  *** construct regression matrix ***

%
%  phi  = regression matrix. 
%  R    = phi'*phi square matrix nxn.

   phi=zeros(length(index),nt);
   ss = 0;

%
%  Autoregressive terms: 

   for kl=1:na,
      phi(:,kl)=-z(index-kl,1);
   end, ss=ss+na;

%
%  Exogenous terms:

   for ku=1:nu         
      for ii=1:nb( ku),
         uf = filterdw( [zeros(nk+ii-1,1) 1], nb(ku)+nk(ku)-1, 1, 0, z(:,2), M, 1, T);
         phi(:,ss+ii) = uf(index); 
      end, ss = ss + nb(ku);
   end

%
%  Include bias term:

   if isempty( bias) ~= 1,
      phi(:,ss+1) = ones(length(index),1);
   end

   R = phi'*phi;
   F = phi'*z(index,1);

%  *** compute estimate ***

   t=phi\z(index,1);

%
%  Write out results:

   if na > 0, f = [1 t(1:na)']; else; f = 1; end

   maxofnb = max(nb);
   if maxofnb > 0,  
      b = NaN*ones(nu, maxofnb); 
      ss  = na;
      for ku=1:nu
         b( ku, 1: nb( ku)) = t(ss+1:ss+nb(ku))'; 
         ss = ss + nb(ku);
      end
   else
      b = [];
   end

%
%  Bias term:

   if isempty( bias) ~=   1,
      bias = t(nt);
   end

   e = z(index,1) - phi*t;

   LAM = e'*e/length(index);
 
   FPE  = LAM * (1+nt/Ncap)/(1-nt/Ncap); %Akaike's FPE 
   BIC  = log( LAM)+nt/2*log(Ncap)/Ncap;
   Pcap = LAM * inv( R);
