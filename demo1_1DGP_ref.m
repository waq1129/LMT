% demo1_1DGP.m
%
% Tutsidal script illustrating P-GPLVM for 1-dimensional latent variable
% with tuning curves generated from 1D Gaussian Process.

% Initialize paths
initpaths;

% Load data
datasetname = 'simdatadir/simdata1.mat';  % name of dataset
if ~exist(datasetname,'file') % Create simulated dataset if necessary
    fprintf('Creating simulated dataset: ''%s''\n', datasetname);
    mkSimData1_1DGP;
end
load(datasetname);
xxtrue = simdata.latentVariable;
yytrue = simdata.spikes;
fftrue = simdata.spikeRates;

% Get sizes and spike counts
[nt0,nneur] = size(yytrue); % nt: number of time points; nneur: number of neurons
nf = size(xxtrue,2); % number of latent dimensions

%% == 1. Compute baseline estimates ====

% Initialize the log of spike rates with the square root of spike counts.
ffmat = sqrt(yytrue);

% % Compute LLE
xlle = lle(ffmat,nf,20);
xllemat = align_xtrue(xlle,xxtrue); % align the estimate with the true latent variable.

% Compute PCA
xppca = pca(ffmat,nf);
xppcamat = align_xtrue(xppca,xxtrue); % align the estimate with the true latent variable.

% Compute Poisson Linear Dynamic System (PLDS)
xplds = run_plds(yytrue,nf)';
xpldsmat = align_xtrue(xplds,xxtrue); % align the estimate with the true latent variable.
xinit = xplds;
xinitmat = xpldsmat;

% truncate into segments, re-organize the data
if nt0>=2000
    nt = find_nt(nt0);
else
    nt = nt0;
end
ntr = nt0/nt;
yy_all = permute(reshape(yytrue,nt,ntr,1,[]),[2,3,1,4]);
xinit_all = permute(reshape(xinit,nt,ntr,1,[]),[2,3,1,4]);
sid = 1:ntr;
tid = 1;
yy = reshape(yy_all(sid,tid,:,:),[],nt,nneur);
xinit = reshape(xinit_all(sid,tid,:,:),[],nt,nf);
if sum(vec(abs(xinit)))==0
    xinit = randn(size(xinit))*1e-5;
end

seg_id = zeros(length(sid),length(tid));
for ii=1:length(sid)
    for jj=1:length(tid)
        seg_id(ii,jj,:) = tid(jj);
    end
end
seg_id = vec(seg_id);

%% == 2. Compute P-GPLVM ====
% Set up options
setopt.sepx_flag = 0;
setopt.sigma2_init = 10;
setopt.sigma2_end = min([0.1,setopt.sigma2_init]); % initial noise variance
setopt.lr = 0.95; % learning rate
setopt.latentTYPE = 1; % kernel for the latent, 1. AR1, 2. SE
setopt.ffTYPE = 2; % kernel for the tuning curve, 1. AR1, 2. SE, 3. linear,4. SE_len
setopt.initTYPE = 2; % initialize latent: 1. use PLDS init; 2. use random init; 3. true xxtrue
setopt.la_flag = 1; % 1. no la; 2. standard la; 3. decoupled la,  obsolete
setopt.rhoxx = 100; % rho for Kxx
setopt.lenxx = 100; % len for Kxx
setopt.rhoff = 1; % rho for Kff
setopt.lenff = median(vec(range(xinit))); % len for Kff
setopt.lenff_ratio = 1; % len ratio for Kff
setopt.b = 0; % obsolete
setopt.r = 1; % obsolete
setopt.nsevar = 1; % obsolete
setopt.hypid = [2,3,4]; % 1. rho for Kxx; 2. len for Kxx; 3. rho for Kff; 4. len for Kff; 5. sigma2 (annealing it instead of optimizing it)
setopt.xinit = xinit; % for initialization purpose
setopt.niter = 500; % number of iterations
setopt.opthyp_flag = 0;

%% == 3. Plot latent variables and tuning curves ====
% Initialize the log of spike rates with the square root of spike counts.

% Get sizes and spike counts
[ntr,nt,nneur] = size(yy); % nt: number of time points; nneur: number of neurons

%
latentTYPE = setopt.latentTYPE; % kernel for the latent, 1. AR1, 2. SE
ffTYPE = setopt.ffTYPE; % kernel for the tuning curve, 1. AR1, 2. SE
xinit = setopt.xinit;

% generate grid values as inducing points
tgrid = [1:nt]';

% set initial noise variance for simulated annealing
lr = setopt.lr; % learning rate
sigma2_init = setopt.sigma2_init;
propnoise_init = 0.001;
sigma2 = sigma2_init;
propnoise = propnoise_init;
b = setopt.b;
r = setopt.r;
nsevar = setopt.nsevar;

% set initial prior kernel
% K = Bfun(eye(nt),0)*Bfun(eye(nt),0)';
% Bfun maps the white noise space to xxtrue space
rhoxx = setopt.rhoxx; % marginal variance of the covariance function the latent xxtrue
lenxx = setopt.lenxx; % length scale of the covariance function for the latent xxtrue
rhoff = setopt.rhoff; % marginal variance of the covariance function for the tuning curve ff
lenff = setopt.lenff; % length scale of the covariance function for the tuning curve ff
lenff_ratio = setopt.lenff_ratio;

% lenxx = 50; % init value, smooth for easy optimization
lenff_ratio = 0.1; % init value

% set hypers
hypers = [rhoxx; lenxx; rhoff; lenff]; % rho for Kxx; len for Kxx; rho for Kff; len for Kff

[Bfun, BTfun, nu] = prior_kernel_sp(rhoxx,lenxx,nt,latentTYPE,tgrid);
Bfun = @(x,f) permute(reshape(Bfun(reshape(permute(x,[2,1,3]),size(x,2),[]),f),[],size(x,1),size(x,3)),[2,1,3]);
BTfun = @(x,f) permute(reshape(BTfun(reshape(permute(x,[2,1,3]),size(x,2),[]),f),[],size(x,1),size(x,3)),[2,1,3]);

% initialize latent
initTYPE = setopt.initTYPE;
switch initTYPE
    case 1  % use LLE or PPCA or PLDS init
        uu0 = Bfun(xinit,1);
    case 2   % use random init
        uu0 = Bfun(xinit,1);
        uu0 = randn(size(uu0))*0.01;
    case 3   % true xxtrue
        uu0 = Bfun(xinit,1)+randn(nu,nf);
end
uu = uu0;  % initialize sample
xxsamp = Bfun(uu,0);
xxsamp_old = xxsamp;
xxsamp_mt = reshape(xxsamp,[],nf);

xxsampmat = reshape(permute(xxsamp,[2,1,3]),[],nf);%align_xtrue(xxsamp,xxtrue);
xxsampmat = align_xtrue(xxsampmat,xxtrue);
xxsampmat_old = xxsampmat;

% grid for inducing points
switch nf
    case 1
        ng = 500;
    case 2
        ng = 25;
    case 3
        ng = 10;
    case 4
        ng = 5;
    case 5
        ng = 5;
    case 6
        ng = 4;
    case 7
        ng = 3;
    case 8
        ng = 2;
end
switch nf
    case 1
        xgrid = gen_grid([min(xxsamp_mt(:,1)) max(xxsamp_mt(:,1))],ng,nf); % x grid (for plotting purposes)
    case 2
        xgrid = gen_grid([min(xxsamp_mt(:,1)) max(xxsamp_mt(:,1)); min(xxsamp_mt(:,2)) max(xxsamp_mt(:,2))],ng,nf); % x grid (for plotting purposes)
    case 3
        xgrid = gen_grid([min(xxsamp_mt(:,1)) max(xxsamp_mt(:,1)); min(xxsamp_mt(:,2)) max(xxsamp_mt(:,2)); min(xxsamp_mt(:,3)) max(xxsamp_mt(:,3))],ng,nf); % x grid (for plotting purposes)
    case 4
        xgrid = gen_grid([min(xxsamp_mt(:,1)) max(xxsamp_mt(:,1)); min(xxsamp_mt(:,2)) max(xxsamp_mt(:,2)); min(xxsamp_mt(:,3)) max(xxsamp_mt(:,3));...
            min(xxsamp_mt(:,4)) max(xxsamp_mt(:,4))],ng,nf); % x grid (for plotting purposes)
    case 5
        xgrid = gen_grid([min(xxsamp_mt(:,1)) max(xxsamp_mt(:,1)); min(xxsamp_mt(:,2)) max(xxsamp_mt(:,2)); min(xxsamp_mt(:,3)) max(xxsamp_mt(:,3));...
            min(xxsamp_mt(:,4)) max(xxsamp_mt(:,4)); min(xxsamp_mt(:,5)) max(xxsamp_mt(:,5))],ng,nf); % x grid (for plotting purposes)
    case 6
        xgrid = gen_grid([min(xxsamp_mt(:,1)) max(xxsamp_mt(:,1)); min(xxsamp_mt(:,2)) max(xxsamp_mt(:,2)); min(xxsamp_mt(:,3)) max(xxsamp_mt(:,3));...
            min(xxsamp_mt(:,4)) max(xxsamp_mt(:,4)); min(xxsamp_mt(:,5)) max(xxsamp_mt(:,5)); min(xxsamp_mt(:,6)) max(xxsamp_mt(:,6))],ng,nf); % x grid (for plotting purposes)
    case 7
        xgrid = gen_grid([min(xxsamp_mt(:,1)) max(xxsamp_mt(:,1)); min(xxsamp_mt(:,2)) max(xxsamp_mt(:,2)); min(xxsamp_mt(:,3)) max(xxsamp_mt(:,3));...
            min(xxsamp_mt(:,4)) max(xxsamp_mt(:,4)); min(xxsamp_mt(:,5)) max(xxsamp_mt(:,5)); ...
            min(xxsamp_mt(:,6)) max(xxsamp_mt(:,6)); min(xxsamp_mt(:,7)) max(xxsamp_mt(:,7))],ng,nf); % x grid (for plotting purposes)
    case 8
        xgrid = gen_grid([min(xxsamp_mt(:,1)) max(xxsamp_mt(:,1)); min(xxsamp_mt(:,2)) max(xxsamp_mt(:,2)); min(xxsamp_mt(:,3)) max(xxsamp_mt(:,3));...
            min(xxsamp_mt(:,4)) max(xxsamp_mt(:,4)); min(xxsamp_mt(:,5)) max(xxsamp_mt(:,5)); ...
            min(xxsamp_mt(:,6)) max(xxsamp_mt(:,6)); min(xxsamp_mt(:,7)) max(xxsamp_mt(:,7)); ...
            min(xxsamp_mt(:,8)) max(xxsamp_mt(:,8))],ng,nf); % x grid (for plotting purposes)
end
if nf==1
    xavg = squeeze(mean(xxsamp,1))';
else
    xavg = squeeze(mean(xxsamp,1));
end
lenff = median(vec(range(xxsamp_mt)))/lenff_ratio;
covfun = covariance_fun(rhoff,lenff,ffTYPE); % get the covariance function
fgrid = covfun(xgrid,xavg)*pdinv(covfun(xavg,xavg))*(squeeze(mean(yy,1)));
fgrid = fgrid/max(vec(fgrid));

% Now do inference
infTYPE = 1; % 1 for MAP; 2 for MH sampling; 3 for hmc
ppTYPE = 1; % 1 optimization for ff; 2. sampling for ff
la_flag = setopt.la_flag; % 1. no la; 2. standard la; 3. decoupled la
opthyp_flag = setopt.opthyp_flag; % flag for optimizing the hyperparameters
sepx_flag = setopt.sepx_flag;

% set options for minfunc
% opt for f
options = [];
options.Method='scg';
options.TolFun=1e-4;
options.MaxIter = 1e1;
options.maxFunEvals = 1e1;
options.Display = 'off';

% opt for x
options1 = [];
options1.Method='scg';
options1.TolFun=1e-4;
options1.MaxIter = 1e1;
options1.maxFunEvals = 1e1;
options1.Display = 'off';

rs = []; % collect r-squared value for our method
niter = setopt.niter;
clf
for iter = 1:niter
    
    % anneal
    if sigma2>setopt.sigma2_end
        sigma2 = sigma2*lr;  % decrease the noise variance with a learning rate
    end
    
    if lenff_ratio<setopt.lenff_ratio & sigma2>setopt.sigma2_end
        lenff_ratio = lenff_ratio/lr;
        lenff = mean(vec(range(xxsamp_mt)))/lenff_ratio;
        hypers(2) = lenff;
    end
    
    if sigma2<=setopt.sigma2_end
        opthyp_flag = 1;
    end
    
    switch nf
        case 1
            xgrid = gen_grid([min(xxsamp_mt(:,1)) max(xxsamp_mt(:,1))],ng,nf); % x grid (for plotting purposes)
        case 2
            xgrid = gen_grid([min(xxsamp_mt(:,1)) max(xxsamp_mt(:,1)); min(xxsamp_mt(:,2)) max(xxsamp_mt(:,2))],ng,nf); % x grid (for plotting purposes)
        case 3
            xgrid = gen_grid([min(xxsamp_mt(:,1)) max(xxsamp_mt(:,1)); min(xxsamp_mt(:,2)) max(xxsamp_mt(:,2)); min(xxsamp_mt(:,3)) max(xxsamp_mt(:,3))],ng,nf); % x grid (for plotting purposes)
        case 4
            xgrid = gen_grid([min(xxsamp_mt(:,1)) max(xxsamp_mt(:,1)); min(xxsamp_mt(:,2)) max(xxsamp_mt(:,2)); min(xxsamp_mt(:,3)) max(xxsamp_mt(:,3));...
                min(xxsamp_mt(:,4)) max(xxsamp_mt(:,4))],ng,nf); % x grid (for plotting purposes)
        case 5
            xgrid = gen_grid([min(xxsamp_mt(:,1)) max(xxsamp_mt(:,1)); min(xxsamp_mt(:,2)) max(xxsamp_mt(:,2)); min(xxsamp_mt(:,3)) max(xxsamp_mt(:,3));...
                min(xxsamp_mt(:,4)) max(xxsamp_mt(:,4)); min(xxsamp_mt(:,5)) max(xxsamp_mt(:,5))],ng,nf); % x grid (for plotting purposes)
        case 6
            xgrid = gen_grid([min(xxsamp_mt(:,1)) max(xxsamp_mt(:,1)); min(xxsamp_mt(:,2)) max(xxsamp_mt(:,2)); min(xxsamp_mt(:,3)) max(xxsamp_mt(:,3));...
                min(xxsamp_mt(:,4)) max(xxsamp_mt(:,4)); min(xxsamp_mt(:,5)) max(xxsamp_mt(:,5)); min(xxsamp_mt(:,6)) max(xxsamp_mt(:,6))],ng,nf); % x grid (for plotting purposes)
        case 7
            xgrid = gen_grid([min(xxsamp_mt(:,1)) max(xxsamp_mt(:,1)); min(xxsamp_mt(:,2)) max(xxsamp_mt(:,2)); min(xxsamp_mt(:,3)) max(xxsamp_mt(:,3));...
                min(xxsamp_mt(:,4)) max(xxsamp_mt(:,4)); min(xxsamp_mt(:,5)) max(xxsamp_mt(:,5)); ...
                min(xxsamp_mt(:,6)) max(xxsamp_mt(:,6)); min(xxsamp_mt(:,7)) max(xxsamp_mt(:,7))],ng,nf); % x grid (for plotting purposes)
        case 8
            xgrid = gen_grid([min(xxsamp_mt(:,1)) max(xxsamp_mt(:,1)); min(xxsamp_mt(:,2)) max(xxsamp_mt(:,2)); min(xxsamp_mt(:,3)) max(xxsamp_mt(:,3));...
                min(xxsamp_mt(:,4)) max(xxsamp_mt(:,4)); min(xxsamp_mt(:,5)) max(xxsamp_mt(:,5)); ...
                min(xxsamp_mt(:,6)) max(xxsamp_mt(:,6)); min(xxsamp_mt(:,7)) max(xxsamp_mt(:,7)); ...
                min(xxsamp_mt(:,8)) max(xxsamp_mt(:,8))],ng,nf); % x grid (for plotting purposes)
    end
    
    %% 1. Find optimal ff
    [Bfun, BTfun, nu] = prior_kernel_sp(rhoxx,lenxx,nt,latentTYPE,tgrid);
    Bfun = @(x,f) permute(reshape(Bfun(reshape(permute(x,[2,1,3]),size(x,2),[]),f),[],size(x,1),size(x,3)),[2,1,3]);
    BTfun = @(x,f) permute(reshape(BTfun(reshape(permute(x,[2,1,3]),size(x,2),[]),f),[],size(x,1),size(x,3)),[2,1,3]);
    
    covfun = covariance_fun(rhoff,lenff,ffTYPE); % get the covariance function
    cxx = covfun(xgrid,xgrid);
    invcxx = pdinv(cxx+cxx(1,1)*sigma2*eye(size(cxx)));
    
    for tt=1:length(tid)
        oo = find(seg_id==tid(tt));
        xxsamp_mt = reshape(xxsamp(oo,:,:),[],nf);
        ctx = covfun(xxsamp_mt,xgrid);
        kk = ctx*invcxx;
        lmlifun_poiss = @(ff) StateSpaceModelsofSpikeTrains_ref(ff,reshape(yy(oo,:,:),[],nneur),invcxx,kk);
        ff0 = vec(fgrid);
        floss_ff = @(ff) lmlifun_poiss(ff); % negative marginal likelihood
        %DerivCheck(floss_ff,ff0)
        [ffnew, fval] = minFunc(floss_ff,ff0,options);
        fgrid = reshape(ffnew,[],nneur);
    end
    xxsamp_mt = reshape(xxsamp(1:length(sid),:,:),[],nf);
    ctx = covfun(xxsamp_mt,xgrid);
    ffmat = reshape(ctx*(invcxx*fgrid),[],nt,nneur);
    
    ffmat = reshape(permute(ffmat,[2,1,3]),[],nneur);
    yymat = reshape(permute(yy(1:length(sid),:,:),[2,1,3]),[],nneur);
    
    figure(1)
    [~,yi] = max(sum(yymat,1));
    subplot(313),plot([yymat(:,yi),exp(ffmat(:,yi))]),title(sigma2),legend('true ff','P-GPLVM ff'), axis tight, drawnow
    gg = size(yymat,1);
    tmp = yymat(1:gg,:)./repmat(max(yymat(1:gg,:))+1e-6,gg,1);
    subplot(311), imagesc(tmp')
    ff1 = exp(ffmat);
    tmp = ff1(1:gg,:)./repmat(max(ff1(1:gg,:))+1e-6,gg,1);
    subplot(312), imagesc(tmp')
    drawnow
    
    %% 2. Find optimal latent xxtrue, actually search in u space, xxtrue=K^{1/2}*u
    invkf = invcxx*fgrid;
    if sepx_flag
        for oo=1:length(seg_id)
            uu = Bfun(xxsamp(oo,:,:),1);
            lmlifun = @(u) logmargli_gplvm_se_block_ref_nogrid(u,xgrid,invkf,reshape(yy(oo,:,:),[],nneur),Bfun,covfun,nf,BTfun,length(oo));
            %DerivCheck(lmlifun,vec(uu))
            uunew = minFunc(lmlifun,vec(uu),options1);
            uu = reshape(uunew,length(oo),[],nf);
            xxsamp(oo,:,:) = Bfun(uu,0);
        end
    else
        for tt=1:length(tid)
            oo = find(seg_id==tid(tt));
            uu = Bfun(xxsamp(oo,:,:),1);
            lmlifun = @(u) logmargli_gplvm_se_block_ref_nogrid(u,xgrid,invkf,reshape(yy(oo,:,:),[],nneur),Bfun,covfun,nf,BTfun,length(oo));
            %DerivCheck(lmlifun,vec(uu))
            uunew = minFunc(lmlifun,vec(uu),options1);
            uu = reshape(uunew,length(oo),[],nf);
            xxsamp(oo,:,:) = Bfun(uu,0);
        end
    end
    
    % plot latent xxtrue
    xxsamp1 = xxsamp;
    if numel(size(xxsamp1))==2
        xxsampmat0 = xxsamp1;
        if nf==1
            xxsampmat0 = xxsampmat0';
        end
    else
        xxsampmat0 = reshape(permute(xxsamp1,[2,1,3]),[],nf);%align_xtrue(xxsamp,xxtrue);
    end
    xxsampmat = align_xtrue(xxsampmat0,xxtrue);
    figure(2),clf
    for dd = 1:nf
        subplot(nf,1,dd)
        plot(1:nt0,xxtrue(:,dd),'m-',1:nt0,xinitmat(:,dd),'g-',1:nt0,xxsampmat(:,dd),'k-',1:nt0,xxsampmat_old(:,dd),'k:','linewidth',2);
    end
    axis tight
    legend('true x','init x','P-GPLVM x','P-GPLVM old x');
    title(iter)
    drawnow
    
    xxsamp_mt = reshape(xxsamp,[],nf);
    xxsampmat_old = xxsampmat;
    xxsamp_old = xxsamp;
    rs = [rs; norm(xxtrue-xxsampmat)];
    
    %% optimze hyperparameters
    if opthyp_flag
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute initial negative log-likelihoods
        hypid = setopt.hypid; % 1. rho for Kxx; 2. len for Kxx; 3. rho for Kff; 4. len for Kff; 5. sigma2 (simulated annealing instead of optimization)
        loghyp0 = log([vec(hypers); sigma2]);
        loghyp = log([rhoxx;lenxx;rhoff;lenff;sigma2]);
        loghyp = loghyp(hypid);
        lmlifun_hyp = @(loghyp) logmargli_gplvm_se_sor_hyp_ref(loghyp,loghyp0,xxsamp,reshape(yy,[],nneur),xgrid,latentTYPE,tgrid,nt,hypid,sigma2,fgrid,ffTYPE,length(seg_id));
        opts = optimset('largescale', 'off', 'maxiter', 1e1, 'display', 'off');
        lb = [-10;0;-10;-10;-10]; % lower bound
        lb = lb(hypid);
        ub = [10;10;10;10;10]; % upper bound
        ub = ub(hypid);
        loghypnew = fmincon(lmlifun_hyp,vec(loghyp),[],[],[],[],lb,ub,[],opts);
        % loghypnew = fminunc(lmlifun_hyp,vec(loghyp),opts);
        loghyp0new = loghyp0;
        loghyp0new(hypid) = loghypnew;
        rhoxx = exp(loghyp0new(1));
        lenxx = exp(loghyp0new(2));
        rhoff = exp(loghyp0new(3));
        lenff = exp(loghyp0new(4));
        sigma2 = exp(loghyp0new(5));
    end
    
    display(['iter:' num2str(iter) ', rs:' num2str(rs(end)) ', rhoxx:' num2str(rhoxx) ', lenxx:' num2str(lenxx) ', rhoff:' num2str(rhoff) ', lenff:' num2str(vec(lenff)') ', lenff_ratio:' num2str(lenff_ratio) ', sigma2:' num2str(sigma2)])
    
end







