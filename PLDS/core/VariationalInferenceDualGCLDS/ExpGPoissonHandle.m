function [f] = ExpGPoissonHandle(y,OverM_ast,OverV_ast,varargin)
%
% exp-Gpoisson likelihood without base measure
% Yuanjun Gao 2015
p_log = OverM_ast + OverV_ast/2;
p_max = max(max(p_log, [], 2), 0);

p_normalizer = log(sum(exp(bsxfun(@minus, p_log, p_max)), 2) + exp(-p_max)) + p_max; %add 1 for category zero

eta_use = y .* OverM_ast;
eta_use(y == 0) = 0;
eta_use = sum(eta_use, 2);

f = - eta_use + p_normalizer;
