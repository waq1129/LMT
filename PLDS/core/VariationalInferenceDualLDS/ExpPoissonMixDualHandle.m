function [f df] = ExpPoissonMixDualHandle(lam,logPi);
%
% dual of standard exp-poisson likelihood
%

if min(lam)<0
  f  = inf;
  df = nan(size(lam));
else
  loglam = log(lam);
  f  = lam'*(loglam-1-logPi);
  df = loglam-logPi;
end