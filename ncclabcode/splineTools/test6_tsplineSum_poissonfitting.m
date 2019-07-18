% test6_tsplineSum_poissonfitting.m
%
% Script illustrating fitting a transformed sum-of-splines (tsplinesum)
% by maximizing Poisson log-likelihood
%
% Last edited: 7 Apr 2012 (JW Pillow)


% 0. Setup: set transfer function and true nonlinearity

% Set transfer function (forcing spline to take positive values)
g = @logexp1;  % log(1+exp(x))  % transfer function

% Set functions to fit
f1 = @(x)(50*sin(x*2*pi)+20);
f2 = @(x)(30*cos(x*pi));
ff = @(x1,x2)(g(f1(x1)+f2(x2)));


%% 1. Generate some noisy data 
nsamps = 500;  % number of points
signse = .01;

x1 = rand(nsamps,1)*2-1;
x2 = rand(nsamps,1)*2-1;
y = poissrnd(ff(x1,x2));

subplot(231);
xvals1 = -1:.01:1;
plot(x1, y, '.', xvals1, g(f1(xvals1)), 'k');
title('raw data');
xlabel('x1'); ylabel('y');

subplot(234);
xvals2 = -1:.02:1;
plot(x2, y, '.', xvals2, dtbin*g(f2(xvals2)), 'k');
xlabel('x2'); ylabel('y');


%% 3. Set up cubic spline parameters & fit by ML
dbreak = 2*.125; % spacing between break-points
ss1.breaks = -1:dbreak:1; % points of discontinuity ("knots")
ss1.smoothness = 3;  % 1 + (# of times differentiable at breaks)
ss1.extrapDeg = [2,2]; % degree polynomial on each end segment 

ss = {ss1,ss1};

% Solve for ML spline params via linear regression
[fspl,splfuns] = fit_tsplinesum_poisson(y,[x1 x2],ss,g);


%% 4. Make Plots

subplot(232);
plot(xvals1, g(f1(xvals1)), 'k', xvals1, g(splfuns{1}(xvals1)), 'r', ...
    ss1.breaks, g(splfuns{1}(ss1.breaks)), 'ro');
title('func 1 fit');
ylabel('g(f1(x1))');
xlabel('x1');

subplot(235);
plot(xvals2, f2(xvals2), 'k', xvals2, splfuns{2}(xvals2), 'r', ...
    ss1.breaks, splfuns{2}(ss1.breaks), 'ro');
title('func 2 fit');
xlabel('x2');
ylabel('g(f2(x2))');

subplot(233);
plot(splfuns{1}(x1), y,'ro'); axis equal; axis tight;
ylabel('y');
xlabel('g(f1(x1))');

subplot(236);
plot(y, y, 'k', fspl([x1 x2]),y,  'ro'); axis equal; axis tight;
ylabel('y');
xlabel('g(f1(x1)+f2(x2))');

%  Note that DC of each separate function cannot be determined, but 
%  the sum f1(x) + f2(x) accurately reproduces y.  Here things look good
%  for the individual function fits because f1 has a mean different from 0,
%  but f2 has mean of zero.

