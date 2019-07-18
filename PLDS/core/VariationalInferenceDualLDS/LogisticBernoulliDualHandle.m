function [f df] = LogisticBernoulliDualHandle(lam,varargin);
%
% dual of standard exp-poisson likelihood
%

if (min(lam)<0)||(max(lam)>1)
  f  = inf;
  df = nan(size(lam));
else
  loglam = log(lam);
  f  = lam'*loglam+(1-lam)'*(log(1-lam));
  df = log(lam./(1-lam));
end
