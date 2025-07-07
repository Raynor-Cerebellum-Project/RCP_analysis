function [b,f,p,Vcap,BIC,Pcap,v,w1,w2]=oew(z,ix,nn,p,FT,maxit,tol,lim,maxsize,Tsamp)
%OEW	Computes the prediction error estimate of an output-error model.
%
%	[b,f,p]=oew(z,ix,nn,p,FT)
%
%	[b,f,p]:Returned as the estimated parameters of the output-error model
%		y(t) = [B(q)/F(q)] u(t-nk) + P/F(q) ic + e(t)
%	Z : 	The output-input data Z=[y u], with y and u column vectors.
%		This routine only works for single-input systems. 
%	NN: 	Initial value and structure information, given either as
%		NN=[nb nf nk f], the orders and delay of the above model and 
%		an estimate of the denominator.
%
%	Some parameters associated with the algorithm are accessed by
%	[b,f,p,MSE,BIC,P,v,w,ic] = oew(z,ix,nn,p,FT,maxit,tol,lim,maxsize,T)
%

%	L. Ljung 10-1-86
%	Copyright (c) 1986 by the MathWorks, Inc.
%	All Rights Reserved.

%   Claudio G. Rey, 10:50AM  9/17/93

% *** Set up default values ***

  [cmptr,maxsdef] = computer;
  if maxsdef < 8192, maxsdef=4096; else maxsdef=256000;end
  if nargin<10, Tsamp=1;end
  if nargin<9, maxsize=maxsdef;end
  if nargin<8, lim=1.6;end
  if nargin<7, tol=0.0001;, end
  if nargin<6, maxit=10;, end

  if Tsamp<0,Tsamp=1;end
  if maxsize<0, maxsize=maxsdef;end
  if lim<0, lim=1.6;end
  if tol<0, tol=0.01;end
  if maxit<0, maxit=10;end

%  *** Do some consistency tests ***

  [nr,cc]=size(nn); [Ncap,nz]=size(z(ix,:)); nu=nz-1;
  N = ix(length(ix)); z = z(1:N,:); [N,nz] = size(z);
  M = ix2mx(ix); ns = length(M(:,1));
  if nz>Ncap,
     error('The data should be organized in column vectors')
  return,end
  if nu>1,
     error('This routine only works for single-input systems. Use PEMW instead!')
  return,end
  if nu==0,
     error('OEW makes sense only if an input is present!')
  return,end

%*** if nn has more than 1 row, then it is a theta matrix
%     and we jump directly to the minimization ***

  if nr>1 nu=nn(1,3);
     if nu>1,
        error('This routine only works for single-input systems')
     return,end
     nf=nn(1,8); nb=nn(1,5); nk=nn(1,9); nt=nf+nb;
     t=nn(3,1:nt)'; nn = [nb,nf,nk];

     if nf>0; f=[1 t(nb+1:nt)'];else f=1;end
     b = [zeros(1,nk) t(1:nb)'];

%    Compute the prediction errors:

     [v,w1,w2,p] = peoew(b,nb+nk-1,f,nf,z,M,FT,p);
% keyboard
  end

  if nr==1
     nb=nn(1); nf=nn(2); nk=nn(3);ni=max(nf,nb+nk-1);nt=nb+nf;
     if cc<3+nf,
     error('This routine requires an estimate of the poles')
     return, end
     pole = nn(4:length(nn));

%    Compute initial estimate

     [b, f, p, v, w1, w2] = pkoew(z,ix,pole,nn,FT,M);

     if nf>0, t=[b(nk+1:nb+nk)'; f(2:nf+1)'];end
  end

  ramp = (1:Ncap)'*Tsamp;

  if FT<3, n = nf + nb + ns*nf; else, n = nf + nb; end

%  Prediction of the output:

   w = w1 + w2;

%  Expanded t vector:

   if FT<3, for i=1:ns, t(nt+1+(i-1)*nf:nt+i*nf) = p(i,:)';end, end

%  Compute the residual power:

   Vcap=v(ix)'*v(ix)/(length(ix));

%  *** Display initial estimate ***

   clc, disp(['  INITIAL ESTIMATE'])
   disp(['Current loss:' num2str(Vcap)])
   disp(['theta'])
   theta=t(1:(nb+nf))

%  *** start minimizing ***

%  ** determine limit for robustification **

   if lim~=0, lim=median(abs(v(ix)-median(v(ix))))*lim/0.7;end
   if lim==0, lim=1000*Vcap;end
   vl = zeros(N,1);
   ll = ones( N,1)*lim;
   vl(ix)=max(min(v(ix),ll(ix)),-ll(ix)); clear ll
   g=ones(n,1); l=0; st=0; nmax=max(nf,nb+nk-1);

%  ** the minimization loop **

   while [norm(g)>tol l<maxit st~=1]

      l=l+1;

%     * compute gradient *

      w1f = -pefiltw( 1, nb+nk-1, f, nb+nk-1,     w1, M, FT);
      w2f = -pefiltw( 1,      nf, f,      nf,     w2, M,  1);
      wf  =  w1f + w2f;
      uf  =  pefiltw( 1, nb+nk-1, f, nb+nk-1, z(:,2), M, FT);
%keyboard
      if FT<3,
         icf =  zeros(N,ns);
         for k = 1:ns,
            If(1:M(k,2),k) = iw(1,f,nf,M(k,:));
         end
      end

      Mem=floor(maxsize/n);
      R=zeros(n);Fcap=zeros(n,1);
      for k1=0:Mem:Ncap-1,
         jj=ix(k1+1:min(Ncap,k1+Mem));
         psi=zeros( length(jj), n);
         o = 0;
         for j=1:nb, psi(:,j+o) = uf(jj-j-nk+1); end; o = o + nb;
         for j=1:nf, psi(:,j+o) = wf(jj-j)     ; end; o = o + nf;
         if FT<3,
            for k=1:ns,
               for j=1:nf,
                  psi(:,j+o) = If(jj-j+1,k);
               end, o=o+nf;
            end,
         end
         if Ncap>Mem,R=R+psi'*psi; Fcap=Fcap+psi'*vl(jj);end
      end
      if Ncap>Mem, g=R\Fcap; else g=psi\(vl(jj));end

%     * search along the g-direction *

      [t1,p1,w1,w2,vl,v,f,V1,st]=soew(z,ix,t,g,nn,ns,FT,lim,Vcap,M);
      w = w1 + w2;

      home
      disp(['      ITERATION # ' int2str(l)])
      disp(['Current loss: ' num2str(V1) '  Previous loss: ' num2str(Vcap)])
      disp(['Current   previous   gn-dir'])
      theta=[t1(1:(nb+nf)) t(1:(nb+nf)) g(1:(nb+nf))]
      disp(['Norm of gn-vector: ' num2str(norm(g))])

      if st==1,
         disp('No improvement of the criterion possible along the search direction'),
         disp('Iterations therefore terminated'),
      end

      t=t1;if FT<3, p=p1; end, Vcap=V1;

   end

   b( 1: nb) = t(1:nb)'; 

   if nf > 0, f = [1 t(nb+1:nb+nf)']; else; f = 1; end
   
   Vcap = v(ix)'*v(ix)/length(ix);

   FPE  = Vcap*(1+n/Ncap)/(1-n/Ncap);
   BIC  = log(Vcap)+n/2*log(Ncap)/Ncap;

   if maxit==0, return,end
   if Ncap>M, PP=inv(R); else PP=inv(psi'*psi);end
   th(4:3+nt,1:nt)=Vcap*PP(1:nt,1:nt);
   Pcap = Vcap*PP(1:nt,1:nt);


