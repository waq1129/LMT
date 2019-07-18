function [f df] = ExpGPoissonDualHandle(lam, varargin),
%
% dual of standard exp-GPoisson likelihood
% Yuanjun Gao 2015

lam0 = 1 - sum(lam,2);
if min(lam(:))<0 || min(lam0(:)) < 0
  f  = inf;
  df = nan(size(lam));
else
  loglam = log(lam); loglam(lam == 0) = 0;
  loglam0 = log(lam0); loglam0(lam0 == 0) = 0;
  f  = lam(:)'*loglam(:) + lam0'*loglam0;
  df = bsxfun(@minus, loglam, loglam0);
end
