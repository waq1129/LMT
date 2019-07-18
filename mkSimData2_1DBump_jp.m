% mkSimData2_1DBump.m
%
% Generates a simulated dataset with a 1D latent trajectory and 1D tuning curves generated from
% 1D Gaussian bumps.
%%
% Initialize paths (and create 'simdatadir' folder if necessary)
initpaths

fname = 'simdatadir/simdata2.mat'; % pathname of file to create

%% 1. Generate latent variables
nt = 1000;  % number of time points
nf = 1; % number of latent dimension

nc = length(nt); % the dimension of the input
gridends = [ones(nc,1) nt(:)]; % range of function to consider
tgrid = gen_grid(gridends,nt,1); % generate the time grid

kAR1 = @(rho,len,x,z) covAR1iso([log(len); log(rho)/2], x, z); % autoregressive lap-1 kernel
kSE = @(rho,len,x,z) covSEiso([log(len); log(rho)/2], x, z, 0, 1); % squared exponential (RBF) kernel

latentTYPE = 1; % Latent Type: 1=AR1, 2=SE or RBF
switch latentTYPE
    case 1, % AR1
        len1 = 25; % length scale
        rho1 = 1; % marginal variance
        Kpriortrue = kAR1(rho1,len1,tgrid,tgrid); % Latent covariance
        xx = mvnrnd(zeros(nt,1),Kpriortrue,nf)'; % Generate latent
        for nn=1:nf
            xx(xx(:,nn)<-6,nn) = -6;
            xx(xx(:,nn)>6,nn) = 6;
        end
    case 2, % SE or RBF
        len1 = 5; % length scale
        rho1 = 1; % marginal variance
        Kpriortrue = kSE(rho1,len1,tgrid,tgrid); % Latent covariance
        xx = mvnrnd(zeros(nt,1),Kpriortrue,nf)'; % Generate latent
end

%% 2. Create neural tuning curves
nneur = 20; % number of neurons
ffTYPE = 3; % Tuning Curve Type: 1=AR1; 2=RBF; 3=gaussianBumps; 4=sin
switch ffTYPE
    case 1, % AR1 cov function for GP
        len2 = 5; % length scale
        rho2 = 1; % marginal variance
        covfun_true = @(x) kAR1(rho2,len2,x,x); % cov fun
        ffun = @(x) exp(mvnrnd(zeros(1,length(x)),covfun_true(x),nneur)');
    case 2, % RBF cov function for GP
        len2 = 1; % length scale
        rho2 = 2; % marginal variance
        covfun_true = @(x) kSE(rho2,len2,x,x); % cov fun
        ffun = @(x) exp(mvnrnd(zeros(1,length(x)),covfun_true(x),nneur)');
    case 3, % gaussian-bump tuning curves
        len2 = 0.4; % standard deviation
        rho2 = 4.75; % peak height
        %ctrs = gen_grid([-6 6;-6 6],nneur*2,nf); % generate the center of the bump
        %ii = size(ctrs,1);
        %jj = randperm(ii);
        %ctrs = ctrs(jj(1:nneur),:);
        ctrs = sort(rand(nneur,1)*7-3.5);
        ffun = @(x)(kSE(rho2,len2,x,ctrs)+.25); % Gaussian bump function
        ffTYPE = 2; % set the type to be SE for inference
    case 4, % sinusoid function
        phi = rand(nneur,1)*2*pi;
        Omega = rand(nneur,nf)*5;
        ffun = @(x)(3*sin(bsxfun(@plus,x*Omega',phi'))-2);
end

% Generate tuning curves ff
xgrid = gen_grid([-4 4;-4 4],200,nf); % x grid (for plotting purposes)
fffull = ffun([xx;xgrid]); % generate ff for xx and for grid
ff = fffull(1:nt,:); % sampled ff values
ffgrid = fffull(nt+1:end,:); % for plotting purposes only

%% 3.  Simulate spike train
dtBin = 1; % bin size for representing time (here 1s => 1 Hz stimulus frame rate)
RefreshRate = 1/dtBin; % Refresh rate
yy = poissrnd(ff/RefreshRate); % Poisson spike counts

% Report number of spikes and spike rate
nsp = sum(yy(:))/nneur; % number of spikes
fprintf('Simulation: %d time bins, %d spikes  (%.2f sp/s)\n', nt, nsp, nsp/nt*RefreshRate);

%% 4. Make plots
subplot(221);
plot(xgrid, ffgrid);
title('tuning curves');
xlabel('x');

subplot(222);
plot(1:nt, xx); title('latent x(t)');

subplot(223);
imagesc(cov(yy')); axis image; colorbar
title('covariance y(t)*y(t)^T'); xlabel('time'); ylabel('time');

subplot(224);
imagesc(yy'); colorbar,title('neural respse y(t)');
xlabel('time');

%% 3. Save the dataset
simdata.latentVariable = xx;
simdata.spikes = yy;
simdata.spikeRates = ff;
simdata.tuningCurve = ffgrid;
simdata.xgrid = xgrid;
simdata.tgrid = tgrid;
simdata.RefreshRate = RefreshRate;
simdata.latentTYPE = latentTYPE;
simdata.ffTYPE = ffTYPE;
save(fname, 'simdata');

