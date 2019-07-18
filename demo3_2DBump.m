% demo3_2DBump.m
%
% Tutorial script illustrating P-GPLVM for 2-dimensional latent variable
% with tuning curves generated from 2D Gaussian bumps.

% Initialize paths
initpaths;

% Load data
datasetname = 'simdatadir/simdata3.mat';  % name of dataset
if ~exist(datasetname,'file') % Create simulated dataset if necessary
    fprintf('Creating simulated dataset: ''%s''\n', datasetname);
    mkSimData3_2DBump;
end
load(datasetname);
xx = simdata.latentVariable;
yy = simdata.spikes;
ff = simdata.spikeRates;

% Get sizes and spike counts
[nt,nneur] = size(yy); % nt: number of time points; nneur: number of neurons
nf = size(xx,2); % number of latent dimensions

%% == 1. Compute baseline estimates ====

% Initialize the log of spike rates with the square root of spike counts.
ffmat = sqrt(yy);

% % Compute LLE
% xlle = lle(ffmat',20,nf)';
% xllemat = align_xtrue(xlle,xx); % align the estimate with the true latent variable.

% Compute PPCA
options = fgplvmOptions('ftc');
xppca = genX_ppca(nf, nneur, ffmat, options);
xppcamat = align_xtrue(xppca,xx); % align the estimate with the true latent variable.

% Compute Poisson Linear Dynamic System (PLDS)
xplds = run_plds(yy,nf)';
xpldsmat = align_xtrue(xplds,xx); % align the estimate with the true latent variable.

%% == 2. Compute P-GPLVM ====

% Set up options
setopt.sigma2_init = 2; % initial noise variance
setopt.lr = 0.95; % learning rate
setopt.latentTYPE = 1; % kernel for the latent, 1. AR1, 2. SE
setopt.ffTYPE = 2; % kernel for the tuning curve, 1. AR1, 2. SE
setopt.initTYPE = 1; % initialize latent: 1. use PLDS init; 2. use random init; 3. true xx
setopt.la_flag = 3; % 1. no la; 2. standard la; 3. decoupled la
setopt.rhoxx = 10; % rho for Kxx
setopt.lenxx = 50; % len for Kxx
setopt.rhoff = 10; % rho for Kff
setopt.lenff = 50; % len for Kff
setopt.hypid = [1,2,3,4]; % 1. rho for Kxx; 2. len for Kxx; 3. rho for Kff; 4. len for Kff; 5. sigma2 (annealing it instead of optimizing it)
% setopt.xpldsmat = xppcamat; % for plotting purpose
% setopt.xplds = xppca; % for initialization purpose
setopt.xpldsmat = xpldsmat; % for plotting purpose
setopt.xplds = xplds; % for initialization purpose
setopt.niter = 50; % number of iterations

% Compute P-GPLVM with Laplace Approximation
result_la = pgplvm_la(yy,xx,ff,setopt);

% Compute P-GPLVM with a variational lower bound
% result_va = pgplvm_va(yy,xx,setopt);

%% == 3. Plot latent variables and tuning curves ====
figure(2)
xgrid = gen_grid([min(result_la.xxsampmat(:,1)) max(result_la.xxsampmat(:,1)); min(result_la.xxsampmat(:,2)) max(result_la.xxsampmat(:,2))],50,nf); % x grid for plotting tuning curves
fftc = exp(get_tc(result_la.xxsampmat,result_la.ffmat,xgrid,result_la.rhoff,result_la.lenff));
fftc_true = simdata.tuningCurve;

for ii=1:nneur
    fg = reshape(ffgrid(:,ii),50,[]);
    xg = gen_grid([-6 6],50,1);
    subplot(131),cla,hold on
    contourf(xg,xg,fg)
    plot(xx(:,1),xx(:,2),'g.-')
    hold off
    title(['tuning curve for neuron ' num2str(ii)]);
    xlabel('x');
    set(gca,'fontsize',15)
    
    subplot(132),surf(xg,xg,reshape(fftc(:,ii),50,[])),title('estimated tc')
    set(gca,'fontsize',15)
    subplot(133),surf(xg,xg,reshape(fftc_true(:,ii),50,[])),title('true tc')
    set(gca,'fontsize',15)
    drawnow,pause
end