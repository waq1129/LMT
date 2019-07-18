% test basic logistic regression code on 1D simulated example

% set up filter
nw = 30;
wts = 2*normpdf(1:nw,nw/2,3)';
fnlin = @expfun;
tt = 1:nw;
clf;
plot(tt,wts,'k');
errfun = @(w)(norm(w-wts).^2);  % error function handle

% Make stimuli & simulate response
nstim = 500;
stim = 1*(randn(nstim,nw));
xproj = stim*wts;
pp = fnlin(xproj);
y = poissrnd(pp);
fprintf('mean rate = %.1f (%d spikes)\n', sum(y)/nstim, sum(y));

% Compute linear regression solution
wls = stim\y;
wls = wls/norm(wls)*norm(wts); % normalize so vector norm is correct
plot(tt,wts,'k',tt,wls);


%% Find ML estimate using Newton-Raphson optimization
lfunc = @(w)(neglogli_poissGLM(w,stim,y,fnlin)); % neglogli function handle

opts = struct('tolX',1e-8,'tolFun',1e-8,'maxIter',1e4,'verbose',0);
tic;
[wml,nlogli,H] = fminNewton(lfunc,wls,opts);
toc;

plot(tt,wts,'k',tt,[wls,wml]);

% % % Check accuracy of grad / Hessian
% neglogliVals = [lfunc(wts),lfunc(wls)];
% HessCheck(lfunc,wts)
%
% % Compare to performance of fminunc
% tic;
% opts2 = optimset('gradobj', 'on', 'Hessian', 'on');
% [wml2,nlogli2] = fminunc(lfunc,wls,opts2);
% toc;


%% Find MAP estimate
hyperprs = [10;.99];  % precision and AR1 parameter


mstruct.neglogli = @neglogli_poissGLM; % neg log-likelihood function
mstruct.logprior = @logprior_AR1;
mstruct.liargs = {stim,y,fnlin}; % args for likelihood function
mstruct.priargs = {}; % additional prior arguments
lfpost = @(w)(neglogpost_GLM(w,hyperprs,mstruct)); % posterior
% HessCheck(lfpost,wls);  % check gradient & Hessian

tic;
[wmap,nlogpost,H] = fminNewton(lfpost,wls*.1,opts);
toc;

plot(tt,wts,'k',tt,[wls,wml,wmap]);
axis tight;
legend('true','LS','ML','MAP');
ebr = 3*sqrt(diag(inv(H)));
hold on;
errorbar(tt,wmap,ebr,'r');
hold off


%% Search 2D space of hyperparameters 
% (compare evidince with filter error)

rhovals = [1 10 100 1000]';  % prior precision
avals = [.8 .9 .95 .975 .99 .999]'; % AR1 smoothness

[hprsMax,wmapMax,maxlogev,evids] = gridsearch_GLMevidence(wmap,mstruct,rhovals,avals);
plot(tt,wts,'k',tt,[wls,wml,wmap,wmapMax]);
axis tight;
legend('true','LS','ML','MAP','MAPgrid');


%% Set hyperparameters by maximizing marginal log-likelihood (EB)

% First, maximize log-evidence for precision and correlation parameters
hh0 = hprsMax;
[wEB,hprsEB,logevid] = findEBestimate_GLM(wmapMax,hh0,mstruct);


% Second, fix correlation parameter to very close to 1, and just maximize
% evidence for precision
alpha = .995;
mstruct2 = mstruct;
mstruct2.logprior = @logprior_AR1fix;
mstruct2.priargs = {alpha};
[wEB2,hprsEB2,logevid2] = findEBestimate_GLM(wEB,hh0(1),mstruct2);


%%  Examine results

Errs = [errfun(wml), errfun(wmapMax), errfun(wEB), errfun(wEB2)]
Evidences = [max(evids(:)), logevid,logevid2]


plot(tt,wts,'k',tt,[wls,wml,wEB,wEB2]);
axis tight;
legend('true','LS','ML','EB1','EB2');
