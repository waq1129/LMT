% test4_tspline_msefitting.m
%
% Simple script illustrating parametrization of a cubic spline and
% fitting by mimimizing MSE

addpath ../nlfuns/

% 1. Set function to fit
g = @logexp1;
f = @(x)g(10*sin(x*2*pi));

% 2. Generate some noisy data 
nsamps = 500;  % number of points
signse = 1;
x = rand(nsamps,1)*2-1+.1;
y = f(x)+randn(nsamps,1)*signse;

subplot(221);
xvals = -1:.01:1.1;
plot(x, y, '.', xvals, f(xvals), 'k');
title('raw data');

%% 3. Set up cubic spline parameters
dbreak = .4;
breaks = [-1.02 -1.01 -1:dbreak:1 1.01 1.02]; % points of discontinuity
smoothness = 3;  % 1+# of times differentiable at breaks
extrapDeg = [1,1]; % degree polynomial on each end segment 
ss = struct('breaks',breaks, 'smoothness', smoothness, 'extrapDeg', extrapDeg,'tfun',g);


%% 4. Fit spline by minimizing MSE
[fspline,pp] = fit_tspline_mse(x,y,ss);

%% 5. ---- Make Plots ---------
ax(2) = subplot(222);
plot(xvals, f(xvals), 'k', xvals, fspline(xvals), 'r', ...
    breaks, fspline(breaks), 'ro');
title('true fun vs. spline fit');

xvals2 = -1.3:.01:1.3; % Examine extrapolation
ax(1) = subplot(224);
plot(xvals2, f(xvals2), 'k', xvals2, fspline(xvals2), 'r', ...
    breaks, fspline(breaks), 'ro');
title('Extrapolation behavior');
axis tight;
linkaxes(ax,'x');