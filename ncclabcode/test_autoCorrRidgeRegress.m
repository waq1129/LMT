% test_autoCorrRidgeRegress.m - short script to illustrate 
% EB ridge regression and EB correlated-ridge regression

% Set up filter
nx = 100;  % filter dimensionality
k = randn(nx,1);  % make random filter
k = gsmooth(k,4);  % smooth filter

% If desired, create some discontinuities
strtInds = [1 21 51]';  % indices where filter is discontinuous
kjump = 1;
ii = [strtInds;nx+1];
for j = 1:length(strtInds);
    k(ii(j):ii(j+1)-1) = flipud(k(ii(j):ii(j+1)-1))+j*kjump;
end
k = k-mean(k); % center k
k = k./norm(k); % make a unit vector

% Plot initial k
t = 1:nx;
clf; plot(t,k,'.-');

% Make error function
err = @(khat)(sum((k-khat).^2));

%% Make stimulus and response

nsamps = 250; % number of stimulus sample
signse = 1;   % stdev of added noise
x = gsmooth(randn(nx,nsamps),1)'; % stimulus (smooth)
y = x*k + randn(nsamps,1)*signse;  % output variable 

% plot to examine noise level
plot(x*k, x*k, 'k.', x*k, y, 'r.');

%%  Compute ML and ridge regression estimates

xdat.xx = x'*x;
xdat.xy = x'*y;
xdat.yy = y'*y;
xdat.ny = nsamps;

% maximum-likelihood estimate
kml = xdat.xx\xdat.xy;  

% Automatic ridge
[kridge,alpha1,nsevar1] = autoRidgeRegress(xdat.xx, xdat.xy, xdat.yy, xdat.ny);

% Automatic correlated-ridge
[kcr1,hyperprs1,CinvCR1] = autoCorrRidgeRegress(xdat);

% Automatic correlated-ridge, with information about independent blocks
[kcr2,hyperprs2,CinvCR2] = autoCorrRidgeRegress(xdat,strtInds);

% ----------------------------------------------------
% Display results: include kml
clf;
showML = 0;
if showML 
    subplot(211);
    plot(t, k,'k',t,kml,'b',t,kridge,'g',t,kcr1,'r',t,kcr2,'c--');
    legend('true', 'ml', 'ridge','cr1','cr2', 'location', 'southeast');
    errs = [err(kml) err(kridge) err(kcr1) err(kcr2)]
else
    subplot(211);
    plot(t, k,'k',t,kridge,'b',t,kcr1,'r',t,kcr2,'c--');
    legend('true','ridge','cr1','cr2', 'location', 'southeast');
    errs = [err(kridge) err(kcr1) err(kcr2)]
end

subplot(223);
imagesc(inv(CinvCR1)); title('prior cov CR1');
subplot(224);
imagesc(inv(CinvCR2)); title('prior cov CR2');

