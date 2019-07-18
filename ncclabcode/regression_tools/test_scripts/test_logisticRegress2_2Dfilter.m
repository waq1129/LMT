% test empirical logistic regression on 2D simulated example

% make 2D filter
nx = 12;
sig = 2.5;
[xx,yy] = meshgrid(1:nx,1:nx);
wtsim = exp(-((xx-nx/2).^2+(yy-nx/2).^2)/(2*sig^2));
clf; subplot(221);
imagesc(wtsim);

% make weights into column vector
wts = wtsim(:);  
nw = length(wts);
errfun = @(w)(norm(w-wts).^2);  % error function
tt = 1:nw;
subplot(222);
plot(tt,wts,'k');

%% Make stimuli & simulate response
nstim = 1000;
stim = 2*(randn(nstim,nw));  % Gaussian stimuli
xproj = stim*wts;
pp = logistic(xproj);
y = rand(nstim,1)<pp;

% Compute linear regression solution
wls = stim\y;
wls = wls/norm(wls)*norm(wts);
plot(tt,wts,'k',tt,wls);

%% Find ML estimate using Newton-Raphson
lfunc = @(w)(neglogli_bernoulliGLM(w,stim,y)); % neglogli function handle
opts = struct('tolX',1e-8,'tolFun',1e-8,'maxIter',1e4,'verbose',0);
tic;
[wml,nlogli,H] = fminNewton(lfunc,wls*.1,opts);
toc;

plot(tt,wts,'k',tt,[wls,wml]);
errs = [errfun(wls), errfun(wml)]

%% Find MAP estimate
hyperprs = [.1;.99];  % precision and AR1 parameter

mstruct.neglogli = @neglogli_bernoulliGLM; % neg log-likelihood function
mstruct.logprior = @logprior_AR1_2D;
mstruct.liargs = {stim,y}; % args for likelihood function
mstruct.priargs = {[nx,nx]}; % log-prior function
lfpost = @(w)(neglogpost_GLM(w,hyperprs,mstruct)); % posterior
% HessCheck(lfpost,wls);  % check gradient & Hessian

tic;
[wmap,nlogpost,H] = fminNewton(lfpost,wls*.1,opts);
toc;

plot(tt,wts,'k',tt,[wls,wml,wmap]);
axis tight;
legend('true','LS','ML','MAP');

% % Compare to performance of fminunc
% tic;
% opts2 = optimset('gradobj', 'on', 'Hessian', 'on');
% [wmap2,nlogpost2] = fminunc(lfpost,wls,opts2);
% toc;

% evaluate log-evidence
logevid_GLM(wmap,hyperprs,mstruct)

%% Search 2D space of hyperparameters 
% (compare evidince with filter error)

rhovals = [1 10 100 1000]';  % prior precision
avals = [.75 .8 .9 .95 .975 .99 .999]'; % AR1 smoothness

[rr,aa] = meshgrid(rhovals,avals);

errs = zeros(size(rr));
evids = zeros(size(rr));
wmaps = zeros(nw,numel(rr));

for jj = 1:numel(rr)
    hh = [rr(jj);aa(jj)];
    lfpost = @(w)(neglogpost_GLM(w,hh,mstruct));
    [wmap,nlogpost,H] = fminNewton(lfpost,wls*.1,opts);
    errs(jj) = errfun(wmap);
    evids(jj) = logevid_GLM(wmap,hh,mstruct);

    % Plot results
    tt = 1:nw;
    plot(tt,wts,'k',tt,[wls*norm(wts)/norm(wls),wmap]);
    axis tight;
    legend('true','LS','MAP');
    drawnow;
    wmaps(:,jj) = wmap;
    
end
subplot(321);
imagesc(log(rhovals),avals,-errs);
title('negative error');ylabel('alpha');xlabel('log precision');
subplot(322);
imagesc(log(rhovals),avals,evids);
title('log-evidence');ylabel('alpha');xlabel('log precision');


%% Set hyperparameters by maximizing marginal log-likelihood (EB)

% First, maximize log-evidence for precision and correlation parameters
hh0 = [1;.9];
[wEB,hprsEB,logevid] = findEBestimate_GLM(wls*.1,hh0,mstruct);



%% Second, fix correlation parameter to very close to 1, and just maximize
% evidence for precision

alpha = .95;
mstruct2 = mstruct;
mstruct2.logprior = @logprior_AR1_2Dfix;
mstruct2.priargs = {mstruct.priargs{:}, alpha};
[wEB2,hprsEB2,logevid2] = findEBestimate_GLM(wEB,hh0(1),mstruct2);



%%

Errs = [min(errs(:)), errfun(wEB), errfun(wEB2)]
Evidences = [max(evids(:)), logevid,logevid2]

subplot(324);
plot(tt,wts,'k',tt,[wls,wEB,wEB2]);
axis tight;
%legend('true','ML','EB1','EB2');

subplot(323);
imagesc(reshape(wts,nx,nx));
subplot(325);
imagesc(reshape(wEB,nx,nx));
subplot(326);
imagesc(reshape(wEB2,nx,nx));
