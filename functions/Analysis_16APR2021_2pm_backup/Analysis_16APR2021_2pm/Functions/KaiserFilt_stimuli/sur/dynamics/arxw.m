function [b,f,LAM,BIC,Pcap]=arxw(z,index,nn,maxsize,Tsamp,p);
%arxw	Computes LS-estimates of segmented domain ARX-models
%
%	Returnes the estimated parameters of the ARX model
%	y(t) = B(q) u(t-nk) + (1-F(q)) y(t) + e(t)
%	along with estimated covariances and structure information.
%
%       z : the output-input data z=[y u], with y and u as column vectors.
%       z=y only.
%
%	INDEX : marks the data that is actually to be used for the
%	identification
%
%	NN: NN = [na nb nk], the orders and delays of the above model.
%	For a time series, NN=na only.
%
%	Some parameters associated with the algorithm are accessed by
%
%	[b,f,bias,MSE,BIC,P] = arxw(z,ix,NN,maxsize,T)
%
%	L. Ljung 10-1-86
%	Copyright (c) 1986 by the MathWorks, Inc.
%	All Rights Reserved.

%   Modified ARX: 11:22AM  9/17/93 - Claudio, G. Rey


%  Ncap = number of data points.
%  nz   = number of inputs + 1 (# of outputs)

  [Ncap,nz]= size(z); nu=nz-1;
  Ncap     = length(index);
  if nz>Ncap,
  error('Data should be organized in column vectors!'),
  return,end
  if length(nn)~=1+2*(nz-1),
  disp(' Incorrect number of orders specified!'),
  disp('For an AR-model nn=na'),
  disp('For an ARX-model, nn=[na nb nk]'),
  error('see above'),
  return,end

%  na = order of regression
%  nb = order of FIR for each input
%  nk = delay for each input 
%  nt  = number of parameters.

  if nz>1, na=nn(1);nb=nn(2:1+nu);nk=nn(2+nu:1+2*nu);
  else na=nn(1); nb=0;,nk=0;end
  nt=na+sum(nb);
  if nn==0,return,end

% *** Set up default values **

  [cmptr,maxsdef] = computer;
  if maxsdef < 8192, maxsdef=4096; else maxsdef=256000;end
  if nargin<6, p=1;end
  if nargin<5, Tsamp=1;end
  if nargin<4, maxsize=maxsdef;end
  if Tsamp<0,Tsamp=1;end  
  if maxsize<0, maxsize=maxsdef; end 
  if p<0,p=1;,end

% *** construct regression matrix ***

% nmax = minimun starting point for the data.
% M    = maximum size of array avaliable.
% phi  = regression matrix.
% R    = phi'*phi square matrix nxn.
 
  nmax=max([na+1 nb+nk])-1;
  M=floor(maxsize/nt);
  R=zeros(nt);F=zeros(nt,1);
  for k=0:M:Ncap-1
      jj = index(k+1:min(Ncap,k+M));
      phi=zeros(length(jj),nt);
      for kl=1:na, phi(:,kl)=-z(jj-kl,1);end 
      ss=na;
      for ku=1:nu
           for kl=1:nb(ku), phi(:,ss+kl)=z(jj-kl-nk(ku)+1,ku+1);end 
           ss=ss+nb(ku);
      end
      if Ncap>M | p~=0, R=R+phi'*phi; F=F+phi'*z(jj,1);end
  end

% *** compute estimate ***

  if Ncap>M, th=R\F; else th=phi\z(jj,1);end
  if p==0, return,end

%  proceed to compute loss function and covariance matrix

   t=th; clear th;

   if na > 0, f = [1 t(1:na)']; else; f = 1; end
   
   maxofnb = max(nb);
   if maxofnb > 0,  
%keyboard
      b = NaN*ones(nu, maxofnb); 
      o  = na;
      for ku=1:nu
         b( ku, 1: nb( ku)) = t(o+1:o+nb(ku))'; 
         o = o + nb(ku);
      end

   else
      b = [];
   end

   nt=na+sum(nb);

   Pcap = inv(R);
   LAM  = (z(index,1)'*z(index,1)-F'*Pcap*F) / Ncap;
   Pcap = LAM * Pcap;
   FPE  = LAM * (1+nt/Ncap)/(1-nt/Ncap); %Akaike's FPE 
   BIC  = log(LAM)+nt/2*log(Ncap)/Ncap;
