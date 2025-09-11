function [t1,p1,w1,w2,vl,v,f,V1,st]=soebw(z,ix,t,g,nn,ns,FT,T,lim,Vcap,M)
%SOEBW   searches for lower values of the prediction error criterion.
%
%	[t1,p1,w1,w2,vl,v,f,V1,st]=soew(z,ix,t,g,nn,ns,FT,T,lim,Vcap,M)
%
%	The routine searches for a lower value of the prediction error cri-
%	terion for the output-error model, starting at t, looking in the
%	G-direction. 
%	t1 is returned as the parameters that give a lower value V1.
%	If no lower value is found, ST=1. F is the F-polynomial associated
%	with t1, and w1,w2,vl,v are the filtered data sequences.
%	The routine is to be used as a subroutine to OEBW. See OEBW for
%	an explanation of the other arguments.

%   Claudio G. Rey - 9:50PM  12/28/93

   nb = nn(1); nf = nn(2); nk = nn(3);
 
   offset = length(t)-(FT<3)*nf*ns-nb-nf;
%keyboard
   Length = ix(length(ix));

%
%  Initialize vectors:

   v  = zeros(Length,1); w1 = v; w2 = v; w = v;

%
%  Define limit array

   l=0;k=1;V1=Vcap+1; st=0;
   ll = ones( size( v)); ll = ll*lim;

%
%  loop up to 10 times:

   while [V1 > Vcap l<10],
      t1=t+k*g;
      if l==9,t1=t;end

      if FT<3, 
         nt = nb+nf+offset; 
         for j=1:ns, p1(j,:) = t1(nt+1+nf*(j-1):nt+nf*j)'; end, 
      else
         nt = length( t);
         p1=[];
      end

      b  = [zeros(1,nk) t1(1:nb)'];
 
      f=fstab([1 t1(nb+1:nb+nf)']);t1(nb+1:nb+nf)=f(2:nf+1)';

%
%     Compute estimate of the output:

      y = z(1:Length,1) - t1(nt)*ones(Length,1)*offset;
      u = z(1:Length,2);
      if FT>2, p1=[]; end
      [v,w1,w2,p1] = peoedw( b, f, [y,u], M, FT, T, p1);

%keyboard
%
%     Robustify error estimate and compute robustified norm:

      vl = max( min( v, ll),-ll);
      V1 = v(ix)'*vl(ix)/length(ix);

      home, disp(int2str(l))
      k=k/2; l=l+1; if l==10, st=1;end
   end
end
