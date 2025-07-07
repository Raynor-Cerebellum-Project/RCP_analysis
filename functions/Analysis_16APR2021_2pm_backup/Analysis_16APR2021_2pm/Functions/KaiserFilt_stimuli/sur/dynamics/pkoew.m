function [b,f,p,v,w1,w2] = pkoew(z,ix,pole,nn,FT,M,maxsize,Tsamp)
%PKOEW Computes the prediction error estimate of an output-error model.
%
%	This routine requires that the poles be known beforehand.
%
%	[B,F,P] = pkoew(Z,INDEX,POLE,NN,FT)
%
%	This routine only works for single-input systems.
%
%	Z    :	The output-input data Z=[y u], y and u are column vectors.
%	INDEX:	Domain index.
%	POLE :	Vector containing pole estimates
%	NN   :	Initial value and structure information, given as
%	NN=[nb nf nk], the orders and delay of the above model
%
%	B,F,P:	Estimate of the B/F transfer function and of the initial  
%		conditions estimates
%
%	If M (marker array) is known using it for input makes the algorithm 
%	faster.  M can be read as the fourth output.
%
%	[B,F,P,v,w,ic] = pkoew(Z,INDEX,POLE,NN,FT,M,maxsize,Tsamp)

%   Claudio G. Rey, 10:53AM  7/27/92

% *** Set up default values ***

  [cmptr,maxsdef] = computer;
  if maxsdef < 8192, maxsdef=4096; else maxsdef=256000;end
  if nargin<8, Tsamp=1;end
  if nargin<7, maxsize=maxsdef;end
  if nargin<6, M = ix2mx(ix); end

  if Tsamp<0,Tsamp=1;end 
  if maxsize<0, maxsize=maxsdef;end
  Conts = (FT-floor(FT/2)*2);

%  *** Do some consistency tests ***

%  keyboard
  [nr,cc]=size(nn); [Ncap,nz]=size(z(ix,:)); nu=nz-1; [N,nz] = size(z);
  if nz>Ncap, 
     error('The data should be organized in column vectors')
  return,end
  if nu>1, 
     error('This routine only works for single-input systems!')
  return,end
  if nu==0, 
  error('PKOEW makes sense only if an input is present!')
  return,end

  nb=nn(1); nf=nn(2); nk=nn(3); nt=nb;
  nf = length(pole); ns = length(M(:,1));

  [a,f] = zp2tf([],pole',1);

  if FT<3,

     uf = pefiltw(  1, nb+nk-1, f, nf, z(:,2), M, FT);

     n = nb + nf*ns;
     If = zeros(N,ns);  
     for k = 1:ns,
        If(1:M(k,2),k) = iw(1,f,nf,M(k,:));
     end
%
     Mem=floor(maxsize/n);
     R=zeros(n);F=zeros(n,1);
     for k=0:Mem:Ncap-1,
        jj=ix(k+1:min(Ncap,k+Mem));
        phi=zeros( length(jj), n);
        for j=1:nb, phi(:,j) = uf(jj-j-nk+1);end
        for k=1:ns,
           for j=1:nf, phi(:,nt+(k-1)*nf+j) = If(jj-j+1,k);end
        end
        if Ncap>Mem, R=R+phi'*phi; F=F+phi'*z(jj,1);end
     end
     if Ncap>Mem, th=R\F; else th=phi\z(jj,1);end

     for j=1:ns, p(j,1:nf) = th((j-1)*nf+nb+1:j*nf+nb)'; end

     b = [zeros(1,nk) th(1:nb)'];

     [ v, w1, w2] = peoew( b, nb+nk-1, f, nf, z, M, FT, p); 

  else
     n = nb;
     [ic, p] = icw( f, nf, z(:,1), M, FT);
     y = z(:,1) - ic;

     uf = pefiltw(  1, 0, f, nf, z(:,2), M, 1);

     Mem=floor(maxsize/n);
     R=zeros(n);F=zeros(n,1);
     for k=0:Mem:Ncap-1,
        jj=ix(k+1:min(Ncap,k+Mem));
        phi=zeros( length(jj), n);
        for i=1:nb, phi(:,i) = uf(jj-i-nk+1);end
        if Ncap>Mem, R=R+phi'*phi; F=F+phi'*y(jj);end
     end
     if Ncap>Mem, th=R\F; else th=phi\y(jj);end

     b = [zeros(1,nk) th(1:nb)'];
     [v, w1, w2, p] = peoew( b, nb+nk-1, f, nf, z, M, FT); 
  end

