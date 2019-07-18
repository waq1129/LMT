function p = gcpdf(theta, g, g_has_zero)
% gcpdf get pdf from the generalized Poisson distribution
% see gcrnd for details
%
% Yuanjun Gao, 2015

if nargin < 3 || isempty(g_has_zero),
    g_has_zero = false;
end

g = reshape(g, 1, []);
if ~g_has_zero,
    g = [0, g];
end
theta_siz = size(theta);
if(theta_siz(end) == 1) theta_siz(end) = []; end
K = length(g) - 1;
log_factorial = [0, cumsum(log(1:K))];
p_log = bsxfun(@plus, bsxfun(@times, theta(:), 0:K), g - log_factorial);
p_log = bsxfun(@minus, p_log, max(p_log, [], 2));
p = exp(p_log);
p = bsxfun(@times, p, 1 ./ sum(p, 2));
p = reshape(p, [theta_siz, K+1]);

end
