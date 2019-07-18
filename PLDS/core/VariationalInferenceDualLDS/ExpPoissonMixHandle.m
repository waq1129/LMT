function [f] = ExpPoissonMixHandle(y,OverM_ast,OverV_ast,logPi);
%
% NEGATIVE exp-poisson likelihood
%

[yDim T] = size(y);

Pi = reshape(exp(logPi),yDim,T);
f  = -y.*OverM_ast+exp(OverM_ast+OverV_ast/2).*Pi;