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
opts = struct('maxFunEvals', 100, 'Method', 'newton', 'display', 'full');

% Allocate space for output variables
ngrid = size(hh,1);
wmaps = zeros(length(w0),ngrid);
logevids = zeros(ngrid,1);

% Search grid
wmap_itr = w0; % warm start init
for jj = 1:ngrid
    fprintf('Iteration %d/%d     \n', jj, ngrid);
    hh_itr = hh(jj,:)';  % hyperparameters for this iteration
    lfpost = @(w)(neglogpost_GLM(w,hh_itr,mstruct));
    wmap_itr = minFunc(lfpost,wmap_itr,opts); % do optimization for w_map
    logevids(jj) = -logevid_GLM(wmap_itr,hh_itr,mstruct);
    fprintf('negative Log-evidence: [%g]\n', logevids(jj));
    wmaps(:,jj) = wmap_itr;
end
fprintf('\n');

% Extract outputs
[maxlogev,jjmax] = min(logevids);
wmap = wmaps(:,jjmax);
hprsmax = hh(jjmax,:)';
if nargout >= 4
   logevids = reshape(logevids,size(h1));
end
