function [t1,p1,w1,w2,vl,v,f,V1,st]=soew(z,ix,t,g,nn,ns,FT,lim,Vcap,M)
%SOEW   searches for lower values of the prediction error criterion.
%
%	[t1,p1,w1,w2,vl,v,f,V1,st]=soew(z,ix,t,g,nn,ns,FT,lim,Vcap,M)
%
%	The routine searches for a lower value of the prediction error cri-
%	terion for the output-error model, starting at T, looking in the
%	G-direction. 
%	T1 is returned as the parameters that give a lower value V1.
%	If no lower value is found, ST=1. F is the F-polynomial associated
%	with T1, and W, VL, v are the filtered data sequences.
%	The routine is to be used as a subroutine to OEW. See OEW for
%	an explanation of the other arguments.

%	L. Ljung 10-1-86
%	Copyright (c) 1986 by the MathWorks, Inc.
%	All Rights Reserved.

%   Claudio G. Rey 1991-10-18

  nb = nn(1); nf = nn(2); nk = nn(3);

  Length = ix(length(ix));

% Initialize vectors:

  v  = zeros(Length,1); w1 = v; w2 = v; w = v;

%  Define limit array

   l=0;k=1;V1=Vcap+1; nt=nf+nb; st=0;
   ll = ones(Length,1); ll = ll*lim;

% loop up to 10 times:

  while [V1 > Vcap l<10],
     t1=t+k*g;
     if l==9,t1=t;end

     if FT<3, for j=1:ns, p1(j,:) = t1(nt+1+nf*(j-1):nt+nf*j)'; end, end

     b  = [zeros(1,nk) t1(1:nb)'];

     f=fstab([1 t1(nb+1:nt)']);t1(nb+1:nt)=f(2:nf+1)';

%    Compute estimate of the output:

     if FT>2,
%keyboard
        [v,w1,w2,p1] = peoew( b, nb+nk-1, f, nf, z, M, FT);

     else
        [v,w1,w2] = peoew( b, nb+nk-1, f, nf, z, M, FT, p1);
     end
     w = w1 + w2;

%    Robustify error estimate and compute robustified norm:

     vl = max( min( v, ll),-ll);
     V1 = v(ix)'*vl(ix)/length(ix);

     home, disp(int2str(l))
     k=k/2; l=l+1; if l==10, st=1;end
  end
