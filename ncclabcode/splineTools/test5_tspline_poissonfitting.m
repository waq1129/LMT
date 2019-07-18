% test5_tspline_poissonfitting.m
%
% Simple script illustrating fitting a transformed spline (tspline)
% by maximizing Poisson log-likelihood
%
% Last edited: 7 Apr 2012 (JW Pillow)

% 0. Setup: set transfer function and true nonlinearity

% Set transfer function (forcing spline to take positive values)
g = @logexp1;  % log(1+exp(x))  % transfer function

% Set nonlinearity to fit
f = @(x)g(8*sin(x*2*pi));

%% 1. Generate some noisy data 
nsamps = 500;  % number of points
x = rand(nsamps,1)*2-1;
y = poissrnd(f(x));


clf;
subplot(221);
xvals = -1:.01:1;
plot(x, y, '.', xvals, f(xvals), 'k');
title('raw data');


%% 2. Set up cubic spline parameters
%g = @expfun;  % exp(x)
dbreak = .25;
breaks = [-1.02 -1.01 -1:dbreak:1]; % points of discontinuity
smoothness = 3;  % 1+# of times differentiable at breaks
extrapDeg = [1,2]; % degree polynomial on each end segment 
ss = struct('breaks',breaks, 'smoothness', smoothness, 'extrapDeg', extrapDeg,'tfun',g);


%% 4. Fit spline by minimizing MSE
[fspline,pp] = fit_tspline_poisson(x,y,ss);


%% 5. ---- Make Plots ---------
ax(2) = subplot(222);
plot(xvals, f(xvals), 'k', xvals, fspline(xvals), 'r', ...
    breaks, fspline(breaks), 'ro');
title('true fun vs. spline fit');

xvals2 = -1.1:.01:1.1; % Examine extrapolation
ax(1) = subplot(224);
plot(xvals2, f(xvals2), 'k', xvals2, fspline(xvals2), 'r', ...
    breaks, fspline(breaks), 'ro');
title('Extrapolation behavior');
axis tight;
linkaxes(ax,'x');
