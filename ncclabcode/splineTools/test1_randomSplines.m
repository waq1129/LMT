% test1_randomSplines.m
%
% Examine some randomly generated splines to inspect effects of
% smoothness and end-point extrapolation settings.


% Vary the "smoothness" and "extrapDeg" params below
smoothness = 3;  % 1 + # of continuous derivs   (in [0,3])
extrapDeg = [1 1]; % degree of edge polynomials (in [0,3])

% Set breaks (or "knots") of spline
breaks = [0:2:10,10.1];  % Set points where polynomials spliced together

% % Alternative that's kind of interesting: 1 cubic segment with linear extrap
% smoothness = 2;
% extrapDeg = [1 1];
% breaks = [0 .1 9.9 10];

% Compute spline parameter matrix
ss = struct('breaks',breaks,'smoothness',smoothness,'extrapDeg',extrapDeg);
M = mksplineParamMtx(ss); % generate param matrix

% Report # of parameters in spline
ndegfree = size(M,2);
fprintf('Number of degrees of freedom in spline: %d\n',ndegfree);

% Generate random spline in this basis
xx = breaks(1)-2:.1:breaks(end)+2; % Fine mesh for examining interpolation
nprs = size(M,2);  % # parameters needed to specify this spline
x = M*randn(nprs,1);
f = mksplinefun(breaks,x); % returns function handle to spline

% Plot spline and knot locations
clf;
subplot(221);
plot(xx, f(xx), '.b-', breaks, f(breaks), 'r*');
title(sprintf('Smoothness=%d;  L=deg%d, Right=deg%d',smoothness,extrapDeg(1),extrapDeg(2)));
ylabel('f(x)');

subplot(222);
df = finitediff(f(xx));
dfbreaks = interp1(xx,df,breaks,'nearest');
plot(xx, df, '.b-', breaks, dfbreaks, 'r*');
title('derivatives');
ylabel('df/dx');

subplot(224);
ddf = finitediff(df);
ddfbreaks = interp1(xx,ddf,breaks,'nearest');
plot(xx, ddf, '.b-', breaks, ddfbreaks, 'r*');
title('2nd derivatives');
ylabel('d^2f/dx^2');
