function [b,f,bias,q,LAM,BIC,Pcap]=arxvw(z,index,nn,np,Ts);
%arxvw   Computes LS-estimates of ARX-models
%
%       [b,f,bias,q] = arxvw(z,ix,NN,NP)
%
%	Returnes the estimated parameters of the ARX model
%	y(t) = B(q) u(t-nk) + (1-F(q)) y(t) + bias + u(t-nk)^2 + e(t) 
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
%	[b,a,bias,MSE,FPE,P] = arxvw(z,ix,NN,maxsize,T)
%

%	L. Ljung 10-1-86
%	Copyright (c) 1986 by the MathWorks, Inc.
%	All Rights Reserved.


%  Modified ARX: Claudio, G. Rey - 9:15PM  1/11/94

%  Ncap = number of data points.
%  nz   = number of inputs + 1 (# of outputs)

   Pcap = [];

   [Ncap,nz]= size(z); nu=nz-1;
   Ncap     = length(index);  
   if nz>Ncap,
      error('Data should be organized in column vectors!'),  
      return,
   end
   if length(nn)~=1+2*(nz-1),  
      disp(' Incorrect number of orders specified!'),
      disp('For an AR-model nn=na'),
      disp('For an ARX-model, nn=[na nb nk]'),
      error('see above'), return,
   end

%  na = order of regression 
%  nb = order of FIR for each input
%  nk = delay for each input  
%  nt  = number of parameters.

   if nz>1,
      na=nn(1);nb=nn(2:1+nu);nk=nn(2+nu:1+2*nu);  
   else 
      na=nn(1); nb=0; nk=0;
   end  
   nt=na+sum(nb)+np;
   if nn==0,return,end


%  *** construct regression matrix ***

%  phi  = regression matrix. 

   phi=zeros(length(index),nt);

   for kl=1:na,
      phi(:,kl)=-z(jj-kl,1);
   end

   ss=na;

   M = ix2mx(index);

   for ku=1:nu         
      for ii=1:nb( ku),
         uf = filterdw( [zeros(nk+ii-1,1) 1], nb(ku)+nk(ku)-1, 1, 0, z(:,2), M, 1, Ts);
         phi(:,ss+ii) = uf(index); 
      end, ss = ss + nb(ku);
   end

   phi(:,ss+1) = ones(length(index),1); ss = ss+1;

   for kp = 2:np
     phi(:,ss+1) = z(index-nk(1),2).^kp; ss = ss + 1;
   end

   R = phi'*phi;
   F = phi'*z(index,1);

%  *** compute estimate ***

%keyboard

   t=phi\z(index,1);

%  proceed to compute loss function and covariance matrix

   if na > 0, f = [1 t(1:na)']; else; f = 1; end

   maxofnb = max(nb);
   if maxofnb > 0,  
      b = NaN*ones(nu, maxofnb); 
      o  = na;
      for ku=1:nu
         b( ku, 1: nb( ku)) = t(o+1:o+nb(ku))'; 
         o = o + nb(ku);
      end
   else
      b = [];
   end

   bias = t(o+1);         o = o + 1;
   q    = t(o+1:o+np-1)'; o = o + np-1;

   LAM=(z(index,1)'*z(index,1)-F'*Pcap*F)/Ncap;

   FPE = LAM*(1+nt/Ncap)/(1-nt/Ncap); %Akaike's FPE 

   Pcap=inv(R);

   LAM=(z(index,1)'*z(index,1)-F'*Pcap*F)/Ncap;

   Pcap=LAM*Pcap;

   BIC  = log( LAM)+nt/2*log(Ncap)/Ncap;
