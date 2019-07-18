function [f] = PoissonBaseMeasure(y,params);
%
% log y!
%

f = -sum(log(gamma(vec(y)+1)));
