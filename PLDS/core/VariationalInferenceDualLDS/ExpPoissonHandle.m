function [f] = ExpPoissonHandle(y,OverM_ast,OverV_ast,varargin);
%
% NEGATIVE exp-poisson likelihood without base measure
%

f = -y.*OverM_ast+exp(OverM_ast+OverV_ast/2);