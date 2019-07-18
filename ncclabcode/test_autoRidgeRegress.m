% test_autoRidgeRegress.m 
%
% Short script to illustrate EB ridge regression and compare to performance
% of maximum likelihood 

% Set up filter
nx = 100;  % filter dimensionality
k = randn(nx,1);  % make filter
k = gsmooth(k,3);
k = k./norm(k)*10;
t = 1:nx;
plot(t,k)

% error function
err = @(khat)(sum((k-khat).^2));

%%  Make stimulus and response

nsamps = 200; % number of stimulus sample
signse = 3;   % stdev of added noise
x = gsmooth(randn(nx,nsamps),1)'; % stimulus (smooth)
y = x*k + randn(nsamps,1)*signse;  % output variable 

% plot to examine noise level
plot(x*k, x*k, 'k.', x*k, y, 'r.');

%%  Compute ML and ridge regression estimates

xx = x'*x;
xy = x'*y;
yy = y'*y;

kml = xx\xy;  % maximum-likelihood estimate
[khat,alphahat,nsevarhat] = autoRidgeRegress(xx,xy,yy,nsamps);

%  Make Plots
clf;
plot(t, k,'k', t, kml, t, khat,'r');
legend('true', 'ML', 'ridge');

% inspect errors
errs = [err(kml) err(khat)]
