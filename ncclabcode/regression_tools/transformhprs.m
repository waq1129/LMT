function htprs = transformhprs(hprs,dir)
% htprs = transformhprs(hprs,dir)
%
% Transform hyper-parameters of Bernoulli GLM to or from R^n for easier
% optimization
%
% INPUTS:
%   hprs [m x 1] - hyperparameters (pre- or post-transform)
%    dir - "+1" => to R^n;  "-1" => from R^n to natural range
%
% OUTPUTS:
%  htprs [m x 1] - transformed (or untransformed) hyperparams
%
% Note: current implementation assumes a log-transform of first parameter and
% logistic transform of remaining params.  Should re-write this to be more
% flexible for other arrangements of hyper-params


if dir == 1
    htprs = [log(hprs(1)); logit(hprs(2:end))];
elseif dir == -1
    htprs = [exp(hprs(1)); logistic(hprs(2:end))];
else
    error('undefined value for "dir" = direction of transform');
end


    
    