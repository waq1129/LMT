function R = gcrnd(theta, g, g_has_zero)
%gcrnd: Generate random arrays from the Generalized count distribution
% P(R = k) \propto exp(theta * k + g(k)) / k!, k = 0,...,length(g)  (g(0) is
% fixed to be 0)
% R = gcrnd(theta, g, g_has_zero)
%
% Arguments:
% theta - a real array of any size, and the output R will
% have the same dimension as theta. (each element of R corresponds to the
% element of theta at the same location)
% g - g function
% g_has_zero - default False meaning g(0) = 0 is fixed and kth element of g vector is g(k); 
%            - if set as True the kth element of g will be g(k-1)
%
%Yuanjun Gao, 2015

if nargin < 3 || isempty(g_has_zero),
    g_has_zero = false;
end

g = reshape(g, 1, []);
if ~g_has_zero,
    g = [0, g];
end
theta_siz = size(theta);
K = length(g) - 1;
nSamp = prod(theta_siz);
log_factorial = [0, cumsum(log(1:K))];
p_log = bsxfun(@plus, bsxfun(@times, theta(:), 0:K), g - log_factorial);
p_log = bsxfun(@minus, p_log, max(p_log, [], 2));
p = exp(p_log);
cdf = cumsum(p, 2);
u = rand(nSamp, 1) .* cdf(:,end);
R = reshape(sum(bsxfun(@gt, u, cdf), 2), theta_siz);

end
