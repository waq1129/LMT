function mu = gcmean(theta, g, g_has_zero)
% gcmean mean of the generalized Poisson distribution
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
K = length(g) - 1;
theta_siz = size(theta);
theta = theta(:);
p = gcpdf(theta, g, true);
mu = sum(bsxfun(@times, p, 0:K), 2);
mu = reshape(mu, theta_siz);
end
