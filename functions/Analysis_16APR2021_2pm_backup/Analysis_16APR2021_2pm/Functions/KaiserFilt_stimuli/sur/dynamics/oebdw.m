function [b,f,o,p,Vcap,BIC,Pcap,v,w1,w2]=oebdw(z,ix,nn,o,p,FT,T,maxit,tol,lim,sparsemethod)
%OEBDW	Computes the prediction error estimate of an output-error model.
%
%       [b,f,o,p]=oebdw(z,ix,nn,o,p,FT,T)
%
%	[b,f,o,p]:	Returned as the estimated parameters of the output-error model
%		y(t) = [B(q)/F(q)] u(t-nk) + P/F(q) ic + O + e(t)
%		along with estimated covariances and structure information.
%	Z : 	The output-input data Z=[y u], with y and u column vectors.
%		This routine only works for single-input systems. 
%		Use PEM otherwise.
%	NN: 	Initial value and structure information, given either as
%		NN=[nb nf nk poles], the orders and delay of the above model and 
%		an estimate of poles of the denominator. 
%	T	Sampling Interval (in sec)
%	FT	Filter type
%			1,3 -> Filter initialized at each segment
%			2,4 -> Filter continued throughout data
%			1,2 -> ICs fitted as parameters
%			3,4 -> ICs extracted from initial data
%
%	Some parameters associated with the algorithm are accessed by
%       [b,f,o,p,Vcap,BIC,Pcap,v,w1,w2] = oebdw(z,ix,nn,o,p,FT,T,maxit,tol,lim)


% (c) Claudio G. Rey - 8:42PM  12/28/93

% *** Set up default values ***
  if nargin<11, sparsemethod = 0;      end
  if nargin<10, lim          = 1.6;    end
  if nargin<9, tol           = 0.0001; end
  if nargin<8, maxit         = 10;     end

  if lim<0,    lim           = 1.6;    end
  if tol<0,    tol           = 0.001;  end
  if maxit<0,  maxit         = 10;     end

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
      error('OEBDW makes sense only if an input is present!')
   return,end

   if cc < 3 + nn(2),
      error('This routine requires an estimate of the poles'),
      return 
   end

%  Check format:

   if nr~=1, error('Wrong format for model description'); end

%
%  Define sizes and initial pole and bias estimates:
   
   nb = nn( 1); nf = nn( 2); nk = nn( 3); np = nb - 1; nt = nb + nf;

   pole = nn( 4 : length( nn));

   if ~isempty( o), if isnan(o), o = 0; end, end

%  Compute initial estimate

   if isempty(o)
      [b, f, dum, p, v, w1, w2] = pkoebdw([z(:,1),z(:,2)],ix,pole,nn,0,FT,T,M,sparsemethod);
   else
      [b, f, dum, p, v, w1, w2] = pkoebdw([z(:,1)-o,z(:,2)],ix,pole,nn,0,FT,T,M,sparsemethod);
   end

   if nf>0, t=[b(nk+1:nb+nk)'; f(2:nf+1)'];end

%
%  Prediction of the output some of zero ic and zero input responses

   w = w1 + w2;

%
%  Bias term:

   if ~isempty( o); nt = nt + 1; t( nt) = o; end

%
%  Expand t vector with initial conditions:

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
   g=ones(n,1); l=0; st=0;

%
%  ** the minimization loop **

   while [norm(g)>tol l<maxit st~=1]

      l=l+1;

%    * compute gradient *

      w1f = -filterdw(  1, nb+nk-1, f, nf,     w1, M, FT, T);
      w2f = -filterdw(  1, nb+nk-1, f, nf,     w2, M,  1, T);
      wf  =  w1f + w2f;

      if FT<3,
         MIf = [];
         for k = 1:ns,
            If = zeros(N,1);
            If(1:M(k,2)) = iw(1,f,nf,M(k,:));
            if sparsemethod
               MIf = [MIf,sparse(If)];
            else
               MIf = [MIf,(If)];
            end
         end
      end

      if ~sparsemethod, psi=zeros( length(ix), n); end

      m=0;
      for ii=1:nb,
         uf = filterdw( [zeros(nk+ii-1,1) 1], nb+nk-1, f, nf, z(:,2), M, FT, T);
         psi(:,m+ii) = uf(ix); 
      end, m = m + nb;

      if sparsemethod, psi = sparse( psi); end

      for j=1:nf, psi(:,j+m) = wf(ix-j); end; m = m + nf;

      if isempty( o) ~= 1,
         psi( :, m + 1) = ones( length( ix),1); m = m + 1;
      end

      if FT<3,
         for k=1:ns,
            for j=1:nf,
               psi(:,j+m) = MIf(ix-j+1,k);
            end, m=m+nf;
         end
      end

      if sparsemethod, 
         R = psi'*psi; F = psi'*vl(ix); g = R\F;
      else
         g = psi \ vl( ix);
      end

%keyboard

%     * search along the g-direction *

      [t1,p1,w1,w2,vl,v,f,V1,st]=soebdw(z,ix,t,g,nn,ns,FT,T,lim,Vcap,M);

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

      t=t1;p=p1;Vcap=V1;

   end

   b( 1: nb) = t(1:nb)'; 

   if nf > 0, f = [1 t(nb+1:nb+nf)']; else; f = 1; end
   
   if ~isempty( o), o = t(nt); end

   Vcap = v(ix)'*v(ix)/length(ix);

   FPE  = Vcap*(1+n/Ncap)/(1-n/Ncap);
   BIC  = log(Vcap)+n/2*log(Ncap)/Ncap;

   PP=inv(psi'*psi);
   Pcap = Vcap*PP(1:nt,1:nt);


