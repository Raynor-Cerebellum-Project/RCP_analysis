function [b,f,bias,LAM,BIC,Pcap,v,w1,w2]=arxodw(z,index,nn,T,sparsemethod);
%arxbow   Computes LS-estimates of ARX-models using a multisegment, multibias fit. 
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
%	NN: 	NN = [na nb nk], the orders and delays of the above model.
%		For a time series, NN=na only.
%	bias :	multibias
%	b,f  :  parameters
%
%	Some parameters associated with the algorithm are accessed by
%
%	[b,a,bias,MSE,BIC,P] = arxodw(z,ix,nn,T,sparsemethod)
%

%  Claudio, G. Rey - 2:04PM  1/15/94

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
   [ns,cc] = size(M);

%
%  na = order of regression 
%  nb = order of FIR for each input
%  nk = delay for each input  
%  nt = number of parameters.
%  ns = number of segments 

   if nz>1,
      na=nn(1);nb=nn(2:1+nu);nk=nn(2+nu:1+2*nu);  
   else 
      na=nn(1); nb=0; nk=0;
   end  
   nt=na+sum(nb)+ns;

%  *** construct regression matrix ***

%
%  phi  = regression matrix. 
%  R    = phi'*phi square matrix nxn.

%   phi=zeros(length(index),nt);
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


   if sparsemethod, phi = sparse(phi); end

%
%  Include bias terms:

   L=length(index); o = 0; 
   for ii = 1: ns,
      if sparsemethod,
         l = M(ii,2)-M(ii,1)+1; 
         iii = sparse(o+1:o+l,ones(1,l),ones(l,1),L,1);
         phi = [phi,iii];
         o = o + l;  
      else
         iii = zeros(L,1);
         l = M(ii,2)-M(ii,1)+1;
         iii(o+1:o+l,1)=ones(l,1);
         phi = [phi,iii];
         o = o + l;
      end
   end, ss = ss+ns;

   R = phi'*phi;
   F = phi'*z(index,1);

%
%  *** compute estimate ***

   if sparsemethod, 
      t = R\F;
   else
      t=phi\z(index,1);
   end

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
%  Save the bias terms:

   bias = t(ss+1:nt);

%
%  Compute the error and the MSE: 

   w1 = phi(:,1:ss)*t(1:ss);

   w2 = phi(:,ss+1:nt)*bias;

   v = z(index,1) - w1 -w2;

   LAM = v'*v/length(index);
 
%
%  Compute other output parameters:

   FPE  = LAM * (1+nt/Ncap)/(1-nt/Ncap); %Akaike's FPE 
   BIC  = log( LAM)+nt/2*log(Ncap)/Ncap;
   Pcap = LAM * inv( R);

