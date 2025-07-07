function [b,nk,f,o,p,v,w,ic] = fitsegi(choice,data,Ts,M,b,nk,f,o,p,FT,option,sp)
%FITSEGI - fit segmented data interactively.
%
%	[B,NK,F,O,P,v,w,ic] = fitsegi( choice, data, Ts, M, B, nk, F, O, P, FT)
%

% (c) Claudio G. Rey - 12:13PM  1/8/94

global flag_dyn flag_ini time est bias

if nargin<7,  b      = 1; f = [1 -.01]; nk = 0; end
if nargin<8,  o      = 0;                      end
if nargin<9,  p      = NaN;                    end
if nargin<10, FT     = 3;                      end
if nargin<11, option = 'v';                    end
if nargin<12, sp     = 0;                      end

f = delta2z( f, Ts); b = b / f(1); f = f / f(1); 

%keyboard

BIC = 0; MSE = 0; bias = ~isempty( o);

ns     = length( M(  :, 1));			% Number of segments
Ncap   =         M( ns,  2);			% Length of data
N      =         4;				

PP = roots( f); np = length( PP); 
ZZ = roots( b); nb = length( b);
GG = b( 1);

n = length(b) + length(f)-1 + bias; P = zeros(n);

if FT >3, nback = max( nb-1, np); else, nback = nb-1; end

if option( 1) == 'b'
   if isempty( o) == 1, choice = 1; else, choice = 3; end 
end

if nk <= 0,
   Z = [data(1:Ncap,3),data((1-nk):(Ncap-nk),2)];
else
   Z = [data(1:Ncap,3),[zeros(nk,1);data(1:(Ncap-nk),2)]];
end

Mx = [ (M( :, 1) + nback), M( :, 2)]; ix = mx2ix( Mx);

if nback > 1,
   izero = mx2ix([M( :, 1), M( :, 1) + nback-1]);
end

%
%  Fit with no bias:

if choice == 1,
   
   if np == 0,
      [b,f,o,MSE,BIC,P]=arxbdw( Z, ix, [np nb 0], [], Ts);
      [v,w,ic,p] = peoedw( b, f, Z, Mx, FT, Ts);
   else
      [b,f,o,p,MSE,BIC,P,v,w,ic] = oebdw(Z,ix,[nb np 0 PP(:)'],[],p,FT,Ts,20,1e-5,1.6,sp);
   end
   
   o = 0;
   if option(1) == 'v'
      if nback>1,
         w( izero) = zeros(length(izero),1);
      end   
      
      %NOTE: this "est" will contain the estimate of fr as plotted on the analysis window
      %      it will be utilized to compute the VAF
      %      the time will containe the corresponding time array         
      
      if (flag_dyn == 1) | (flag_dyn == 9999),
         est = [];    est = (w( 1:ix( length( ix))) + ic( 1:ix(length(ix))) + o);
         time = [];     time = ((1: ix( length( ix)))' * Ts);
      elseif (flag_dyn == 33),
         est = [];    est = (w( 1:ix( length( ix))) + ic( 1:ix(length(ix))) + o);
         time = [];     time = ((1: ix( length( ix)))' * Ts);
         hold on,
         plot( ((1:ix(length(ix)))' * Ts), (w( 1:ix(length(ix))) + ic( 1:ix(length(ix))) + o),'w'),
         hold off,
      else
         hold on,
         plot( ((1:ix(length(ix)))' * Ts), (w( 1:ix(length(ix))) + ic( 1:ix(length(ix))) + o),'w'),
         hold off,
      end
   end
   
   rms = sqrt( v( ix)' * v( ix) / (length( ix))); err = mean( v( ix));  dev = std( v( ix));
   %
   for k = 1:N
      Mxx = round(Mx*[(N-k+1) (N-k);(k-1) k]/N);
      for j = 1:ns, segerr(k,j) = mean( v( Mxx( j,1): Mxx( j,2) ) ); end
      segstd( k) = std( segerr( k, :)); segmean( k) = mean(segerr( k, :));
   end
   overallsegmean = mean( segmean);overallsegstd = std( segmean);
   o = [];
   
   
   %
   %  Fit with bias:
   
elseif choice == 2,
   
   if np == 0,
      [b,f,o,MSE,BIC,P]=arxbdw( Z, ix, [np nb 0], NaN, Ts);
      [v,w,ic,p] = peoedw( b, f, [Z(:,1)-o Z(:,2)], Mx, FT, Ts);
   else
      [b,f,o,p,MSE,BIC,P,v,w,ic] = oebdw(Z,ix,[nb np 0 PP(:)'],o,p,FT,Ts,20,1e-5,1.6,sp);
   end
   
   if option(1) == 'v'
      if nback>1,
         w( izero) = zeros(length(izero),1);
      end  
      
      if (flag_dyn == 1),
         est = [];    est = (w( 1:ix( length( ix))) + ic( 1:ix(length(ix))) + o);
         time = [];     time = ((1: ix( length( ix)))' * Ts);
      elseif (flag_dyn == 33) | (flag_dyn == 9999),
         est = [];    est = (w( 1:ix( length( ix))) + ic( 1:ix(length(ix))) + o);
         time = [];     time = ((1: ix( length( ix)))' * Ts);
         hold on,
         plot( ((1:ix(length(ix)))' * Ts), (w( 1:ix(length(ix))) + ic( 1:ix(length(ix))) + o),'w'),
         hold off,
      else      
         hold on, 
         plot( ((1: ix( length( ix)))' * Ts), (w( 1:ix( length( ix))) + ic( 1:ix(length(ix))) + o), 'w'),
         hold off,
      end;  
   end
   
   rms = sqrt( v( ix)' * v( ix) / (length( ix))); err = mean( v( ix));  dev = std( v( ix));
   for k = 1:N
      Mxx = round(Mx*[(N-k+1) (N-k);(k-1) k]/N);
      for j = 1:ns, segerr(k,j) = mean( v( Mxx( j,1): Mxx( j,2) ) ); end
      segstd( k) = std( segerr( k, :)); segmean( k) = mean(segerr( k, :));
   end
   overallsegmean = mean( segmean);overallsegstd = std( segmean);
   
   %
   %  Poke with no bias:
   
elseif choice == 3,
   
   o = 0;
   [b,f,p,v,w,ic] = pkoebdw(Z,ix,PP(:)',[nb np 0],[],FT,Ts,Mx,sp);
   PP = roots( f); np = length( PP);
   
   if option(1) == 'v'
      if nback>1,
         w( izero) = zeros(length(izero),1);
      end   
      
      if (flag_dyn == 1),
         est = [];    est = (w( 1:ix( length( ix))) + ic( 1:ix(length(ix))) + o);
         time = [];     time = ((1: ix( length( ix)))' * Ts);
      elseif (flag_dyn == 33) | (flag_dyn == 9999),
         est = [];    est = (w( 1:ix( length( ix))) + ic( 1:ix(length(ix))) + o);
         time = [];     time = ((1: ix( length( ix)))' * Ts);
         hold on,
         plot( ((1:ix(length(ix)))' * Ts), (w( 1:ix(length(ix))) + ic( 1:ix(length(ix))) + o),'w'),
         hold off,
      else
         hold on, 
         plot( ((1:ix(length(ix)))' * Ts), (w( 1:ix(length(ix))) + ic( 1:ix(length(ix))) + o), 'w'), 
         hold off,
      end;     
   end
   
   rms = sqrt( v( ix)' * v( ix) / (length( ix))); err = mean( v( ix));  dev = std( v( ix));
   for k = 1:N
      Mxx = round(Mx*[(N-k+1) (N-k);(k-1) k]/N)
      for j = 1:ns, segerr(k,j) = mean( v( Mxx( j,1): Mxx( j,2) ) ); end
      segstd( k) = std( segerr( k, :)); segmean( k) = mean(segerr( k, :));
   end
   overallsegmean = mean( segmean);overallsegstd = std( segmean);
   o = [];
   
   %
   %  Poke with bias
   
elseif choice == 4,
   
   [b,f,o,p,v,w,ic] = pkoebdw(Z,ix,PP(:)',[nb np 0],NaN,FT,Ts,Mx,sp);
   PP = roots( f); np = length( PP);
   
   if option(1) == 'v'
      if nb>1,
         w( izero) = zeros(length(izero),1);
      end   
      
      if (flag_dyn == 1),
         est = [];    est = (w( 1:ix( length( ix))) + ic( 1:ix(length(ix))) + o);
         time = [];     time = ((1: ix( length( ix)))' * Ts);
      elseif (flag_dyn == 33) | (flag_dyn == 9999),
         est = [];    est = (w( 1:ix( length( ix))) + ic( 1:ix(length(ix))) + o);
         time = [];     time = ((1: ix( length( ix)))' * Ts);
         hold on,
         plot( ((1:ix(length(ix)))' * Ts), (w( 1:ix(length(ix))) + ic( 1:ix(length(ix))) + o),'w'),
         hold off,
      else     
         hold on, 
         plot( ((1:ix(length(ix)))' * Ts), (w( 1:ix(length(ix))) + ic( 1:ix(length(ix))) + o), 'w'),
         hold off,
      end     
   end
   
   rms = sqrt( v( ix)' * v( ix) / (length( ix))); err = mean( v( ix));  dev = std( v( ix));
   for k = 1:N
      Mxx = round(Mx*[(N-k+1) (N-k);(k-1) k]/N);
      for j = 1:ns, segerr(k,j) = mean( v( Mxx( j,1): Mxx( j,2) ) ); end
      segstd( k) = std( segerr( k, :)); segmean( k) = mean(segerr( k, :));
   end
   overallsegmean = mean( segmean);overallsegstd = std( segmean);
   
   %
   %  Predict
   
elseif choice == 5,
   
   nb = length( ZZ)+1; np = length( PP);
   
   if ~bias, o = 0; end
   [v,w,ic,p] = peoedw( b, f, [Z(:,1)-o Z(:,2)], Mx, FT, Ts);
   
   if option(1) == 'v'
      if nback>1,
         w( izero) = zeros(length(izero),1);
      end   
      
      if (flag_dyn == 1),
         est = [];    est = (w( 1:ix( length( ix))) + ic( 1:ix(length(ix))) + o);
         time = [];     time = ((1: ix( length( ix)))' * Ts);
      elseif (flag_dyn == 33) | (flag_dyn == 9999),
         est = [];    est = (w( 1:ix( length( ix))) + ic( 1:ix(length(ix))) + o);
         time = [];     time = ((1: ix( length( ix)))' * Ts);
         hold on,
         plot( ((1:ix(length(ix)))' * Ts), (w( 1:ix(length(ix))) + ic( 1:ix(length(ix))) + o),'w'),
         hold off,
      else
         hold, plot( ((1:ix(length(ix)))' * Ts), (w( 1:ix(length(ix))) + ic( 1:ix(length(ix))) + o), 'w'),hold
      end
   end
   
   rms = sqrt( v( ix)' * v( ix) / (length( ix))); err = mean( v( ix));  dev = std( v( ix));
   for k = 1:N
      Mxx = round(Mx*[(N-k+1) (N-k);(k-1) k]/N);
      for j = 1:ns, segerr(k,j) = mean( v( Mxx( j,1): Mxx( j,2) ) ); end
      segstd( k) = std( segerr( k, :)); segmean( k) = mean(segerr( k, :));
   end
   overallsegmean = mean( segmean);overallsegstd = std( segmean);
   
   if ~bias, o = []; end;
   %
   %  Multiple bias
   
elseif choice == 6,
   
   [b,f,p,LAM,BIC,Pcap,v,w,ic]=arxodw(Z,ix,[nb np 0],T,sp);
   PP = roots( f); np = length( PP);
   
   if option(1) == 'v'
      if nb>1,
         w( izero) = zeros(length(izero),1);
      end  
      
      if (flag_dyn == 1),
         est = [];    est = (w( 1:ix( length( ix))) + ic( 1:ix(length(ix))) + o);
         time = [];     time = ((1: ix( length( ix)))' * Ts);
      elseif (flag_dyn == 33) | (flag_dyn == 9999),
         est = [];    est = (w( 1:ix( length( ix))) + ic( 1:ix(length(ix))) + o);
         time = [];     time = ((1: ix( length( ix)))' * Ts);
         hold on,
         plot( ((1:ix(length(ix)))' * Ts), (w( 1:ix(length(ix))) + ic( 1:ix(length(ix))) + o),'w'),
         hold off,
      else     
         hold on, 
         plot( ((1:ix(length(ix)))' * Ts), (w( 1:ix(length(ix))) + ic( 1:ix(length(ix))) + o), 'w'),
         hold off,
      end     
   end
   
   rms = sqrt( v( ix)' * v( ix) / (length( ix))); err = mean( v( ix));  dev = std( v( ix));
   for k = 1:N
      Mxx = round(Mx*[(N-k+1) (N-k);(k-1) k]/N);
      for j = 1:ns, segerr(k,j) = mean( v( Mxx( j,1): Mxx( j,2) ) ); end
      segstd( k) = std( segerr( k, :)); segmean( k) = mean(segerr( k, :));
   end
   overallsegmean = mean( segmean);overallsegstd = std( segmean);
   
end

clear Z

sf = sum( f);
b = b/sf; f = z2delta(f,Ts)/sf;
parameter_std = sqrt( diag( P)');
parameter_std(1:np+nb) = parameter_std(1:np+nb)/sf;

if option( 1) ~= 'b'
   
   m = 0; eq = [];
   
   eq = [eq '[' num2str(b(1)) ' ~ ' num2str( parameter_std(1)) ']'];
   
   for ii = 2:nb
      eq = [eq ' + [' num2str(b(ii)) ' ~ ' num2str( parameter_std(m+ii)) ']'];
      eq = [eq 's^' num2str(ii-1)];
   end, m = m + nb;
   
   eq = [eq ' / 1'];
   
   for ii = 1:np
      eq = [eq ' + [' num2str(f(ii+1)) ' ~ ' num2str( parameter_std(ii+m)) ']'];
      eq = [eq 's^' num2str(ii)];
   end, m = m + np;
   
   %keyboard
   
   if bias,
      eq = [eq ' + ' num2str( o) ' ~ ' num2str( parameter_std(m+1))]; m = m+1; 
   end
   
   bias = o;
   
   disp(' ')
   disp(eq)
   disp(' ')
   PP = roots( f); ZZ = roots( b); GG = b( 1);
   disp(['Gain: ' num2str(GG) ' - N. TC(s): ' numa2str(-ZZ) ' - D. TC(s): ' numa2str(-PP) ' - Delay: ' num2str(nk)]) 
   disp(['Error Mean, std. dev., rms and BIC:  ',  numa2str(  [err dev rms BIC])]) 
   
   disp(' ')
   disp(['Seg mean error: ', numa2str(  segmean) ' -  Seg std dev: ' numa2str(  segstd)]) 
   disp(['Overall seg mean error and std dev:  ', numa2str(  [overallsegmean overallsegstd])]) 
   
   if (flag_dyn ~= 1),
      fitsignals = data;
      parameters_out = [b f(2:length(f)) o parameter_std err BIC];
      
      dynmout(fitsignals,time,est,parameters_out,bias);      
   end
   
   %NOTE: flag_dyn will equal 1 only if SlideMN was called before. In this case
   %      the Slideend function will be called, and the program will go on
   %      if the flag is not equal to one, this section is skipped and dynamic
   %      analysis looks exactly as before
   
   if (flag_dyn == 1),         %when optimization slide with position during saccades
      Slideend;
   end;
   
end;


