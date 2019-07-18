function x = logit(p)
% x = logit(p)
%
% Compute logit function:
%   x = log(p./(1-p));
%
% Input:  p \in (0,1)
% Output: x \in Reals

x  = log(p./(1-p));
