% test empirical logistic regression on 2D simulated example

% make 2D filter
nx = 12;
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
stim = 1.5*(randn(nstim,nw));  % Gaussian stimuli
stim = [stim,ones(nstim,1)];
xproj = stim*[wts;3.2];
pp = logistic(xproj);
y = rand(nstim,1)<pp;

% Compute linear regression solution
wls = stim\y;
wls = wls/norm(wls)*norm(wts);
plot(tt,wts,'k',tt,wls(1:nw));

%% Find MAP estimate

hyperprs = [1;.95];  % precision and AR1 parameter
rhoDC = .01; % prior precision for DC term
opts = struct('tolX',1e-8,'tolFun',1e-8,'maxIter',1e4,'verbose',0);

mstruct.neglogli = @neglogli_bernoulliGLM; % neg log-likelihood function
mstruct.logprior = @logprior_AR1_2D;
mstruct.liargs = {stim,y}; % args for likelihood function
mstruct.priargs = {[nx,nx],rhoDC}; % log-prior function
lfpost = @(w)(neglogpost_GLM(w,hyperprs,mstruct)); % posterior
% HessCheck(lfpost,wls);  % check gradient & Hessian

% do optimization using Newton's method
opts = struct('tolX',1e-8,'tolFun',1e-8,'maxIter',1e4,'verbose',0);
tic;
[wmap,nlogpost,H] = fminNewton(lfpost,wls*.1,opts);
toc;

plot(tt,wts,'k',tt,[wls(1:nw),wmap(1:nw)]);
axis tight;
legend('true','LS','MAP');
DCmap = wmap(end)  % MAP estimate of DC term

%% Search 2D space of hyperparameters 
% (compare evidince with filter error)

rhovals = [1 10 100 1000]';  % prior precision
avals = [.75 .8 .9 .95 .975 .99 .999]'; % AR1 smoothness

[hprsMax,wmap,maxlogev,evids] = gridsearch_GLMevidence(wls*.1,mstruct,rhovals,avals);


%% Set hyperparameters by maximizing marginal log-likelihood (EB)

% First, maximize log-evidence for precision and correlation parameters
hh0 = hprsMax;
[wEB,hprsEB,logevid,Hess] = findEBestimate_GLM(wmap,hh0,mstruct);


%% Second, fix correlation parameter to very close to 1, and just maximize
% evidence for precision

alpha = .95;
mstruct2 = mstruct;
mstruct2.logprior = @logprior_AR1_2Dfix;
mstruct2.priargs = {mstruct.priargs{:}, alpha};
[wEB2,hprsEB2,logevid2,Hess2] = findEBestimate_GLM(wEB,hh0(1),mstruct2);

%% Third, for comparison, look at 1D AR1 prior

alpha = .999;
mstruct3 = mstruct;
mstruct3.logprior = @logprior_AR1fix;
mstruct3.priargs = {alpha,nx^2,rhoDC};
[wEB3,hprsEB3,logevid3,Hess3] = findEBestimate_GLM(wEB,hh0(1),mstruct3);


%%

DCtrms = [wmap(end), wEB(end), wEB2(end), wEB3(end)]
wwls=wls(1:nw);wwmap=wmap(1:nw);
wwEB=wEB(1:nw);wwEB2=wEB2(1:nw);wwEB3=wEB3(1:nw);

Errs = [errfun(wwls), errfun(wwmap), errfun(wwEB), errfun(wwEB2) errfun(wwEB3)]
Evidences = [max(evids(:)), logevid,logevid2,logevid3]

subplot(324);
plot(tt,wts,'k',tt,[wwls,wwEB,wwEB2,wwEB3]);
axis tight;
%legend('true','ML','EB1','EB2');

subplot(221);
imagesc(reshape(wts,nx,nx)); title('true wts');
subplot(223);
imagesc(reshape(wwEB,nx,nx)); title('EB2D');
subplot(224);
imagesc(reshape(wwEB2,nx,nx)); title('EB2Dfix');
subplot(222);
imagesc(reshape(wwEB3,nx,nx)); title('EB1D');
