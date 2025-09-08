function y = pdf(name,x,a,b,c)
%PDF	Computes a chosen probability density function.
%	Y = PDF(NAME,X,A) returns the named probability density
%	function, which uses parameter A, at the the values in X.
%	Y = PDF(NAME,X,A,B,) returns the named probability density
%	function, which uses parameters A and B, at the the values in X.
%	Similarly for Y = PDF(NAME,X,A,B,C).
%
%	The name can be: 'beta' or 'Beta', 'bino' or 'Binomial',
%	'chi2' or 'Chisquare','exp' or 'Exponential', 'f' or 'F', 
%	'gam' or 'Gamma','geo' or 'Geometric','hyge' or 'Hypergeometric',
%	'norm' or 'Normal','poiss' or 'Poisson','t' or 'T','unif' or 'Uniform',
%	'unid' or 'Discrete Uniform','weib' or 'Weibull'.
% 
%	PDF calls many specialized routines that do the calculations. 

%	D. Zwillinger 5-91, B. Jones 10-92
%	Copyright (c) 1993 by The MathWorks, Inc.
%	$Revision: 1.2 $  $Date: 1993/08/25 14:30:11 $
 
if nargin<2, 
    error('PDF requires at least two input arguments.'); 
end

if ~isstr(name), 
    error('PDF requires the first argument to be the name of a distribution.'); 
end

if nargin<5, 
    c = 0; 
end 
if nargin<4, 
    b = 0; 
end 
if nargin<3, 
    a = 0; 
end 

if     strcmp(name,'beta') | strcmp(name,'Beta'),  
    y = betapdf(x,a,b);
elseif strcmp(name,'bino') | strcmp(name,'Binomial'),  
    y = binopdf(x,a,b);
elseif strcmp(name,'chi2') | strcmp(name,'Chisquare'), 
    y = chi2pdf(x,a);
elseif strcmp(name,'exp') | strcmp(name,'Exponential'),   
    y = exppdf(x,a);
elseif strcmp(name,'f') | strcmp(name,'F'),     
    y = fpdf(x,a,b);
elseif strcmp(name,'gam') | strcmp(name,'Gamma'),   
    y = gampdf(x,a,b);
elseif strcmp(name,'geo') | strcmp(name,'Geometric'),   
    y = geopdf(x,a);
elseif strcmp(name,'hyge') | strcmp(name,'Hypergeometric'),  
    y = hygepdf(x,a,b,c);
elseif strcmp(name,'norm') | strcmp(name,'Normal'), 
    y = normpdf(x,a,b);
elseif strcmp(name,'poiss') | strcmp(name,'Poisson'),
    y = poisspdf(x,a);
elseif strcmp(name,'t') | strcmp(name,'T'),     
    y = tpdf(x,a);
elseif strcmp(name,'unid') | strcmp(name,'Discrete Uniform'),  
    y = unidpdf(x,a);
elseif strcmp(name,'unif')  | strcmp(name,'Uniform'),  
    y = unifpdf(x,a,b);
elseif strcmp(name,'weib') | strcmp(name,'Weibull'),  
    y = weibpdf(x,a,b);
else   
    error('Sorry, the statistics toolbox does not support this distribution.'); 
end 
