% test empirical Bayes logistic regression code on simulated 1D example


% 1.  Set up simulated example

% set up filter
nw = 50; % number of coeffs in filter
wts = 3*normpdf(1:nw,nw/2,sqrt(nw)/2)';  % linear filter
b = -1; % constant (DC term)

% Make stimuli & simulate response
nstim = 2000;
stim = 1*(randn(nstim,nw));
xproj = stim*wts+b;
pp = logistic(xproj);
yy = rand(nstim,1)<pp;

% -- make plot ---
tt = 1:nw;
subplot(212);
plot(tt,wts,'k');
title('true filter');
subplot(211);
xpl = min(xproj):.1:max(xproj);
plot(xproj,yy,'.',xpl,logistic(xpl), 'k');
xlabel('input'); ylabel('response');
fprintf('mean rate = %.1f (%d ones)\n', sum(yy)/nstim, sum(yy));

errfun = @(w)(norm(w-wts).^2);  % error function handle


%% 2. Compute (standard) linear regression estimates
xx = [stim, ones(nstim,1)];  % regressors

% LS estimate
wls = xx\yy;
% MAP estimate
lam = 10000; % ridge parameter
wmap0 = (xx'*xx + lam*speye(nw+1))\(xx'*yy);

subplot(212);
plot(tt,wts,'k',tt,wls(1:nw)/norm(wls(1:nw))*norm(wts),...
    tt,wmap0(1:nw)/norm(wmap0(1:nw))*norm(wts));
legend('original', 'LS', 'ridge');

%% 3. Compute Empirical Bayes logistic-regression estimate, ridge prior

rhovals = 10.^(0:6)'; % grid over prior precision (hyperparameter)
rdgInds = (1:nw); % indices to apply ridge parameter to
rhoNull = .01;  % prior precision for other variables
[wRidge,rhoHat,SDerrbars,Hess] = autoRegress_logisticRidge(xx,yy,rdgInds,rhoNull,rhovals,wmap0);

plot(tt,wts,'k'); hold on;
errorbar(tt,wRidge(1:nw),2*SDerrbars(1:nw),'r');
axis tight;
hold off;


%% 4. Compute Empirical Bayes logistic-regression estimate, AR1 prior

rhovals = 10.^(0:6)'; % grid over prior precision (hyperparameter)
avals = [.8 .9 .95 .975 .99 .995]'; % grid over correlation (AR1 hyperparameter)
rhoNull = .01;  % prior precision for other variables
[wAR1,hprsAR1,SDerrbarsAR1,HessAR1] = autoRegress_logisticAR1(xx,yy,nw,rhoNull,rhovals,avals,wmap0);

plot(tt,wts,'k'); hold on;
errorbar(tt,wRidge(1:nw),2*SDerrbars(1:nw),'r');
errorbar(tt,wAR1(1:nw),2*SDerrbarsAR1(1:nw),'c');
axis tight;
hold off;
legend('true', 'ridge', 'AR1');

Errs = [errfun(wRidge(1:nw)), errfun(wAR1(1:nw))]
