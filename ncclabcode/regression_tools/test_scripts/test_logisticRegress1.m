% test basic logistic regression code on 1D simulated example

% set up filter
nw = 30;
wts = 1*normpdf(1:nw,nw/2,3)';
tt = 1:nw;
clf;
plot(tt,wts,'k');
errfun = @(w)(norm(w-wts).^2);  % error function handle

% Make stimuli & simulate response
nstim = 1000;
stim = 2.5*(randn(nstim,nw));
xproj = stim*wts;
pp = logistic(xproj);
y = rand(nstim,1)<pp;

% Compute linear regression solution
wls = stim\y;
wls = wls/norm(wls)*norm(wts); % normalize so vector norm is correct
plot(tt,wts,'k',tt,wls);


%% Find ML estimate using Newton-Raphson optimization
lfunc = @(w)(neglogli_bernoulliGLM(w,stim,y)); % neglogli function handle

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
mstruct.neglogli = @neglogli_bernoulliGLM;
mstruct.logprior = @logprior_AR1;
mstruct.liargs = {stim,y};
mstruct.priargs = {};
lfpost = @(w)(neglogpost_GLM(w,hyperprs,mstruct));
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

logevid_GLM(wmap,hyperprs,mstruct)

%% Search 2D space of hyperparameters 
% (compare evidince with filter error)

rhovals = [1 10 100 1000]';  % prior precision
avals = [.8 .9 .95 .975 .99 .999]'; % AR1 smoothness

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
    plot(tt,wts,'k',tt,[wls,wml,wmap]);
    axis tight;
    legend('true','LS','ML','MAP');
    drawnow;
    wmaps(:,jj) = wmap;
    
end
subplot(221);
imagesc(log(rhovals),avals,-errs);
title('negative error');ylabel('alpha');xlabel('log precision');
subplot(222);
imagesc(log(rhovals),avals,evids);
title('log-evidence');ylabel('alpha');xlabel('log precision');

%% Set hyperparameters by maximizing marginal log-likelihood (EB)

% First, maximize log-evidence for precision and correlation parameters
hh0 = [10;.95];
[wEB,hprsEB,logevid] = findEBestimate_GLM(wls*.1,hh0,mstruct);


%% Second, fix correlation parameter to very close to 1, and just maximize
% evidence for precision
alpha = .995;
mstruct2 = mstruct;
mstruct2.logprior = @logprior_AR1fix;
mstruct2.priargs = {alpha};
[wEB2,hprsEB2,logevid2] = findEBestimate_GLM(wEB,hh0(1),mstruct2);

%%  Examine results
Errs = [errfun(wml), min(errs(:)), errfun(wEB), errfun(wEB2)]
Evidences = [max(evids(:)), logevid,logevid2]

subplot(223);
plot(tt,wts,'k',tt,[wls,wml,wEB,wEB2]);
axis tight;
legend('true','LS','ML','EB1','EB2');
