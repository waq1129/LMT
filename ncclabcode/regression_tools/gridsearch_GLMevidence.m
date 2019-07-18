function [hprsmax,wmap,maxlogev,logevids] = gridsearch_GLMevidence(w0,mstruct,varargin)
% [hprsmax,wmap,maxlogev,logevids] = gridsearch_GLMevidence(w0,mstruct,varargin)
%
% Search 1d, 2d or 3D grid of hyperparameters for maximum evidence (i.e.,
% for initializing gradient ascent of evidence) 
% 
% Maximizes posterior and uses Laplace approximation to evaluate evidence (marginal likelihood) at each grid point  for
%
% INPUTS:
%  wts [m x 1] - regression weights
%  hprs [p x 1] - hyper-parameters
%  mstruct - model structure with fields
%        .neglogli - func handle for negative log-likelihood
%        .logprior - func handle for log-prior 
%        .liargs - cell array with args to neg log-likelihood
%        .priargs - cell array with args to log-prior function
%   h1 [p1 x 1] - grid for 1st hyperparameter
%   h2 [p2 x 1] - grid for 2nd hyperparameter (optional)
%   h3 [p3 x 1] - grid for 3rd hyperparameter (optional)
%
% OUTPUTS:
%  hprsmax - hyperparameters maximizing marginal likelihood on grid
%     wmap - MAP estimate at evidence maximum
% maxlogev - value of log-evidence at maximum
% logevids - log-evidence across entire grid 
%
% $Id$

% Determine how many hyperparameters used
switch length(varargin)
    case 1
	h1 = varargin{1};
	hh = h1(:);
    case 2
	[h1,h2] = meshgrid(varargin{1},varargin{2});
	hh = [h1(:), h2(:)];
    case 3
	[h1,h2,h3] = meshgrid(varargin{1},varargin{2},varargin{3});
	hh = [h1(:), h2(:), h3(:)];
end

% Optimization parameters (for finding MAP estimate)
opts = struct('tolX',1e-8,'tolFun',1e-8,'maxIter',1e4,'verbose',0);

% Allocate space for output variables
ngrid = size(hh,1);
wmaps = zeros(length(w0),ngrid);
logevids = zeros(ngrid,1);

% Search grid
fprintf('\n%d gridpts:',ngrid);
for jj = 1:ngrid
    hh_itr = hh(jj,:)';  % hyperparameters for this iteration
    lfpost = @(w)(neglogpost_GLM(w,hh_itr,mstruct));
    wmap_itr = fminNewton(lfpost,w0,opts);
    logevids(jj) = logevid_GLM(wmap_itr,hh_itr,mstruct);
    wmaps(:,jj) = wmap_itr;

    if mod(jj,5)==0
	fprintf(' %d', jj);
    end
end
fprintf('\n');

% Extract outputs
[maxlogev,jjmax] = max(logevids);
wmap = wmaps(:,jjmax);
hprsmax = hh(jjmax,:)';
if nargout >= 4
   logevids = reshape(logevids,size(h1));
end
