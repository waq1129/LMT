% test empirical logistic regression on 2D simulated example

% make 2D filter
nx = 16;
sig = 2;
[xx,yy] = meshgrid(1:nx,1:nx);
wtsim = 2*exp(-((xx-nx/2).^2+(yy-nx/2).^2)/(2*sig^2));
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

%% Find MAP estimate
hyperprs = [.5;.95];  % precision and AR1 parameter
opts = struct('tolX',1e-8,'tolFun',1e-8,'maxIter',1e4,'verbose',0);

mstruct.neglogli = @neglogli_bernoulliGLM; % neg log-likelihood function
mstruct.logprior = @logprior_AR1_2D;
mstruct.liargs = {stim,y}; % args for likelihood function
mstruct.priargs = {[nx,nx]}; % log-prior function
lfpost = @(w)(neglogpost_GLM(w,hyperprs,mstruct)); % posterior
% HessCheck(lfpost,wls);  % check gradient & Hessian

tic;
[wmap,nlogpost,H] = fminNewton(lfpost,wls*.1,opts);
toc;

plot(tt,wts,'k',tt,[wls,wmap]);
axis tight;
legend('true','LS','MAP');

%% Search 2D space of hyperparameters 
% (compare evidince with filter error)

rhovals = [1 10 100 1000]';  % prior precision
avals = [.75 .8 .9 .95 .975 .99 .999]'; % AR1 smoothness

[hprsMax,wmap,maxlogev,evids] = gridsearch_GLMevidence(wls*.1,mstruct,rhovals,avals);


%% Set hyperparameters by maximizing marginal log-likelihood (EB)

% First, maximize log-evidence for precision and correlation parameters
hh0 = hprsMax+[5;-.4];
[wEB,hprsEB,logevid] = findEBestimate_GLM(wmap,hh0,mstruct);


%% Second, fix correlation parameter to very close to 1, and just maximize
% evidence for precision

alpha = .95;
mstruct2 = mstruct;
mstruct2.logprior = @logprior_AR1_2Dfix;
mstruct2.priargs = {mstruct.priargs{:}, alpha};
[wEB2,hprsEB2,logevid2] = findEBestimate_GLM(wEB,hh0(1),mstruct2);

%% Third, for comparison, look at 1D AR1 prior

alpha = .999;
mstruct3 = mstruct;
mstruct3.logprior = @logprior_AR1fix;
mstruct3.priargs = {alpha};
[wEB3,hprsEB3,logevid3] = findEBestimate_GLM(wEB,hh0(1),mstruct3);

%%

Errs = [errfun(wls), errfun(wmap), errfun(wEB), errfun(wEB2) errfun(wEB3)]
Evidences = [max(evids(:)), logevid,logevid2,logevid3]

subplot(324);
plot(tt,wts,'k',tt,[wls,wEB,wEB2,wEB3]);
axis tight;
%legend('true','ML','EB1','EB2');

subplot(221);
imagesc(reshape(wts,nx,nx)); title('true wts');
subplot(223);
imagesc(reshape(wEB,nx,nx)); title('EB2D');
subplot(224);
imagesc(reshape(wEB2,nx,nx)); title('EB2Dfix');
subplot(222);
imagesc(reshape(wEB3,nx,nx)); title('EB1D');
