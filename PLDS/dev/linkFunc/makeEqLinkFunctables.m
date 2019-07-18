function [ftab ftablog Mus Stds] = makeEqLinkFunctables(linkFunc,loglinkFunc,varargin)
%
% generates tables for E_q[log p(y|x)]
%
%

dMu    = 35;  % no of grid points for mean
dStd   = 30;  % no of grid points for standard deviation

MuMin  = -10; % interval for mean
MuMax  =  10;

StdMin = 0;   % interval for std
StdMax = 10;

assignopts(who,varargin);

Mus  = (1:dMu)/dMu*(MuMax-MuMin)+MuMin;        
Stds = [1e-10 (1:(dStd-1))/(dStd-1)*(StdMax-StdMin)+StdMin];   


ftab = nan(dMu,dStd);
ftablog = nan(dMu,dStd);
for dm=1:dMu
  for ds=1:dStd
    % Gaussian integral over link function
    fint = @(x) normpdf(x,Mus(dm),Stds(ds)).*linkFunc(x);  
    ftab(dm,ds) = integral(fint,Mus(dm)-3*Stds(ds),Mus(dm)+3*Stds(ds));
    % Gaussian integral over log-link function
    fint = @(x) normpdf(x,Mus(dm),Stds(ds)).*loglinkFunc(x);
    ftablog(dm,ds) = integral(fint,Mus(dm)-3*Stds(ds),Mus(dm)+3*Stds(ds));
  end
end
