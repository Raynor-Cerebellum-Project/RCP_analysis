function [b,f,o,p,Vcap,BIC,Pcap,v,w1,w2]=oebw(z,ix,nn,o,p,FT,maxit,tol,lim,maxsize,Tsamp)
%OEBW	Computes the prediction error estimate of an output-error model.
%
%       [b,f,o,p]=oebw(z,ix,nn,o,p,FT)
%
%	[b,f,o,p]:	Returned as the estimated parameters of the output-error model
%		y(t) = [B(q)/F(q)] u(t-nk) + P/F(q) ic + O + e(t)
%		along with estimated covariances and structure information.
%	Z : 	The output-input data Z=[y u], with y and u column vectors.
%		This routine only works for single-input systems. 
%		Use PEM otherwise.
%	NN: 	Initial value and structure information, given either as
%		NN=[nb nf nk f], the orders and delay of the above model and 
%		an estimate of poles of the denominator. 
%
%	Some parameters associated with the algorithm are accessed by
%       [b,f,o,p,Vcap,BIC,Pcap,v,w1,w2] = oebw(z,ix,nn,o,p,FT,maxit,tol,lim,maxsize,T)


% Claudio G. Rey - 4:32PM  11/19/93

% *** Set up default values ***
  [cmptr,maxsdef] = computer;
  if maxsdef < 8192, maxsdef=4096; else maxsdef=256000;end
  if nargin<11, Tsamp=1;end
  if nargin<10, maxsize=maxsdef;end
  if nargin<9, lim=1.6;end
  if nargin<8, tol=0.0001;, end
  if nargin<7, maxit=10;, end

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
      error('This routine only works for single-input systems.')
   return,end
   if nu==0,
      error('OEBW makes sense only if an input is present!')
   return,end

   if nr==1
      nb = nn( 1); nf = nn( 2); nk = nn( 3); nt = nb + nf;
      if cc < 3 + nf,
      error('This routine requires an estimate of the poles')
      return, end
      pole = nn( 4 : length( nn));

%     Compute initial estimate

      [b, f, p, v, w1, w2] = pkoew([z(:,1)-o,z(:,2)],ix,pole,nn,FT,M);
      if nf>0, t=[b(nk+1:nb+nk)'; f(2:nf+1)'];end
   else

      error('Expected a single row in the parameter list ''NN'' ')
   end

%  Prediction of the output:

   w = w1 + w2;

%  Bias term:

   if isempty( o) ~=   1,
      nt = nt + 1;
      if isnan(o) == 1, t( nt) = 0; else, t( nt) = o; end
   end

%  Expanded t vector:

   if FT<3, for i=1:ns, t(nt+1+(i-1)*nf:nt+i*nf) = p(i,:)';end, end

   if FT<3, n = nt + ns*nf; else, n = nt; end

%  Compute the residual power:
 
   Vcap=v(ix)'*v(ix)/(length(ix));

%  *** Display initial estimate ***

   clc, disp(['  INITIAL ESTIMATE'])
   disp(['Current loss:' num2str(Vcap)])
   disp(['Parameter Values'])
   theta=t(1:nt)

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

%    * compute gradient *

      w1f = -pefiltw( 1, nb+nk-1, f, nb+nk-1,     w1, M, FT);
      w2f = -pefiltw( 1,      nf, f,      nf,     w2, M,  1);
      wf  =  w1f + w2f;
      uf  =  pefiltw( 1, nb+nk-1, f, nb+nk-1, z(:,2), M, FT);

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
	 m = 0;
         for j=1:nb, psi(:,j+m) = uf(jj-j-nk+1); end; m = m + nb;
         for j=1:nf, psi(:,j+m) = wf(jj-j)     ; end; m = m + nf;

         if isempty( o) ~= 1,
            psi( :, m + 1) = ones( length( jj),1); m = m + 1;
         end

         if FT<3,
            for k=1:ns,
               for j=1:nf,
		  psi(:,j+m) = If(jj-j+1,k);
               end, m=m+nf;
            end
         end

         if Ncap>Mem,R=R+psi'*psi; Fcap=Fcap+psi'*vl(jj);end
      end
      if Ncap>Mem, g=R\Fcap; else g=psi\vl(jj);end

%     * search along the g-direction *

      [t1,p1,w1,w2,vl,v,f,V1,st]=soebw(z,ix,t,g,nn,ns,FT,lim,Vcap,M);
      w = w1 + w2;

      home
      disp(['      ITERATION # ' int2str(l)])
      disp(['Current loss: ' num2str(V1) '  Previous loss: ' num2str(Vcap)])
      disp(['Current   previous   gn-dir'])
      theta=[t1(1:nt) t(1:nt) g(1:nt)]
      disp(['Norm of gn-vector: ' num2str(norm(g))])

      if st==1,
         disp('No improvement of the criterion possible along the search direction'),
         disp('Iterations therefore terminated'),
      end

      t=t1;if FT<3, p=p1; end, Vcap=V1;

   end

   b( 1: nb) = t(1:nb)'; 

   if nf > 0, f = [1 t(nb+1:nb+nf)']; else; f = 1; end
   
   o    = t(nt);

   Vcap = v(ix)'*v(ix)/length(ix);

   FPE  = Vcap*(1+n/Ncap)/(1-n/Ncap);
   BIC  = log(Vcap)+n/2*log(Ncap)/Ncap;

   if maxit==0, return,end
   if Ncap>M, PP=inv(R); else PP=inv(psi'*psi);end
   th(4:3+nt-1,1:nt-1)=Vcap*PP(1:nt-1,1:nt-1);
   Pcap = Vcap*PP(1:nt,1:nt);


