function [BBw, nu, sdiag, iikeep] = BfromK(Kprior,sigma2,th)
if nargin<3
    th = 1e6;
end
% Prune search space for xx using SVD of prior covariance
[u,s,v] = svd(Kprior+sigma2*eye(size(Kprior))); sdiag = diag(s);
% iikeep = (sdiag-sigma2)>1e-1;
iikeep = sdiag>sdiag(1)/th;
BB = u(:,iikeep); % orthogonal basis
ss = sqrt(sdiag(iikeep)); % eigenvalues we keep
BBw = BB*diag(ss);
nu = size(BBw,2);
sdiag = sdiag(iikeep);
