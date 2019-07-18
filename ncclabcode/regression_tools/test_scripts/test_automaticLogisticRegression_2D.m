% test empirical Bayes logistic regression code on simulated 1D example


% 1.  Set up simulated example

% make 2D filter
nx = 16;
nt = 8;  
sig = 2;  % RF spread
[xx,tt] = meshgrid(1:nx,1:nt);
wtsim = 2*exp(-((xx-nx/2).^2+(tt-1-nt/2).^2)/(2*sig^2)).*cos(xx-nx/2);
wts = wtsim(:);
nw = nx*nt;
b = -1; % constant (DC term)

% Make stimuli & simulate response
nstim = 2000;
stim = 1*(randn(nstim,nw));
xproj = stim*wts+b;
pp = logistic(xproj);
yy = rand(nstim,1)<pp;

% -- make plot ---
clf; 
subplot(221);
imagesc(wtsim);title('true filter');
iiw = 1:nw;
subplot(222);
plot(iiw,wts,'k');
subplot(223);
xpl = min(xproj):.1:max(xproj);
plot(xproj,yy,'.',xpl,logistic(xpl), 'k');
xlabel('input'); ylabel('response');
fprintf('mean rate = %.1f (%d ones)\n', sum(yy)/nstim, sum(yy));

errfun = @(w)(norm(w-wts).^2);  % error function handle

%% 2. Compute linear and ridge regression estimates
xx = [stim, ones(nstim,1)];  % regressors

% LS estimate
wls = xx\yy;  wlsplot = wls(1:nw)./norm(wls(1:nw))*norm(wts);
% MAP estimate
lam = 1000; % ridge parameter
wmap0 = (xx'*xx + lam*speye(nw+1))\(xx'*yy);
wmapplot = wmap0(1:nw)./norm(wmap0(1:nw))*norm(wts);

subplot(212);
plot(iiw,wts,'k',iiw,wlsplot,iiw,wmapplot);
legend('original', 'LS', 'ridge');

%% 3. Compute Empirical Bayes logistic-regression estimate, AR1 2D prior

rhovals = 10.^(0:6)'; % grid over prior precision (hyperparameter)
avals = [.5 .75 .9 .95 .99]'; % grid over correlation (AR1 hyperparameter)
rhoNull = .01;  % prior precision for other variables
[wAR1,hprsAR1,SDerrbarsAR1,HessAR1] = autoRegress_logisticAR1_2D(xx,yy,[nt nx],rhoNull,rhovals,avals,wmap0);

plot(iiw,wts,'k'); hold on;
%errorbar(iiw,wRidge(1:nw),2*SDerrbars(1:nw),'r');
errorbar(iiw,wAR1(1:nw),2*SDerrbarsAR1(1:nw),'c');
axis tight;
hold off;
legend('true', 'ridge', 'AR1');

%Errs = [errfun(wRidge(1:nw)), errfun(wAR1(1:nw))]
Err = [errfun(wAR1(1:nw))]
