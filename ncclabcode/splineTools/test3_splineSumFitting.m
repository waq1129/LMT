% test_SplineSumFitting.m
%
% Illustrates fitting a sum of cubic splines to data with multiple
% regressors

% 1. Set functions to fit
f1 = @(x)sin(x*2*pi);
f2 = @(x)cos(x*.5*pi);

% 2. Generate some noisy data 
nsamps = 256;  % number of points
signse = .01;
x1 = rand(nsamps,1)*2-1;
x2 = rand(nsamps,1)*4-2;
y = f1(x1)+f2(x2)+randn(nsamps,1)*signse;

subplot(231);
xvals1 = -1:.01:1;
plot(x1, y, '.', xvals1, f1(xvals1), 'k');
title('raw data');
xlabel('x1'); ylabel('y');

subplot(234);
xvals2 = -2:.02:2;
plot(x2, y, '.', xvals2, f2(xvals2), 'k');
xlabel('x2'); ylabel('y');

%% 3. Set up cubic spline parameters
dbreak = 2*.125; % spacing between break-points
ss1.breaks = -1:dbreak:1; % points of discontinuity
ss1.smoothness = 3;  % 1 + (# of times differentiable at breaks)
ss1.extrapDeg = [2,2]; % degree polynomial on each end segment 

dbreak = .5; % spacing between break-points
ss2.breaks = -2:dbreak:2; % points of discontinuity
ss2.smoothness = 3;  % 1 + (# of times differentiable at breaks)
ss2.extrapDeg = [2,3]; % degree polynomial on each end segment 

ss = {ss1,ss2};

%% Solve for ML spline params via linear regression
[fspl,splfuns] = fit_splinesum_mse(y,[x1 x2], ss);


%% 4. Make Plots: examine fits
%  Note that DC of each separate function cannot be determined, but 
%  the sum f1(x) + f2(x) accurately reproduces y.

subplot(232);
plot(xvals1, f1(xvals1), 'k', xvals1, splfuns{1}(xvals1), 'r', ...
    ss1.breaks, splfuns{1}(ss1.breaks), 'ro');
title('func 1 fit');
ylabel('y');  xlabel('x1');

subplot(235);
plot(xvals2, f2(xvals2), 'k', xvals2, splfuns{2}(xvals2), 'r', ...
    ss2.breaks, splfuns{2}(ss2.breaks), 'ro');
title('func 2 fit');
xlabel('x2');

subplot(233);
plot(y, splfuns{1}(x1), 'ro'); axis equal; axis tight;
xlabel('y');
ylabel('f1(x1)');

subplot(236);
plot(y, y, 'k', y, fspl([x1 x2]), 'ro'); axis equal; axis tight;
xlabel('y');
ylabel('f1(x1)+f2(x2)');

