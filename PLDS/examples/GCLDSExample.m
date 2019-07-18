% PLDS toolbox example
%
% Yuanjun Gao, Lars Buesing, Jakob H Macke, 2015
%


clear all
close all


% set parameters for the ground truth GCLDS model 
xDim    = 2;												% latent dimensiom
yDim    = 50;											    	% observed dimension = no of neurons
T       = 50;												% no of time bins per trial; here a time step is approx 10ms 
Trials  = 20;		    										% no trials
maxIter = 100;		    										% max no of EM iterations for fitting the model

% different g function corresponds to different observation distribution
g = [-1]; %binary with logistic regression
%g = -1.9 * (1:10); %something close to Poisson
%g = 0.2 * (1:5).^2 - 2.1 * (1:5); %(almost)-convex function: over-dispersed
%g = -0.4 * (1:5).^2 - 1.5 * (1:5); %concave function: under-dispersed

K = length(g); %how large the support to assume
%%%% ground truth model

trueparams = GCLDSgenerateExample('xDim',xDim,'yDim',yDim,'g',g); % create ground truth model parameters
seqOrig    = GCLDSsample(trueparams,T,Trials);								% sample from the model
tp         = trueparams;										

% print out some statistics of the artificial data
fprintf('Max spike count:    %i \n', max(vec([seqOrig.y])))
fprintf('Mean spike count:   %d \n', mean(vec([seqOrig.y])))
fprintf('Freq non-zero bin:  %d \n', mean(vec([seqOrig.y])>0.5))


%% GCLDS initialization
%%%% fitting a model to artificial data

seq    = seqOrig;
params = [];


%params.model.notes.gStatus=0: GCLDS-full, different g'' across neurons
%params.model.notes.gStatus=1: same g for all neurons
%params.model.notes.gStatus=2: GCLDS-simple, same g'' for all neurons
%params.model.notes.gStatus=3: GCLDS-linear, linear g
params.model.notes.gStatus = 3; 

% initialize parameters, options are:
% - Poisson SSID 'PLDSID', see [Spectral learning of linear dynamics from generalised-linear observations with application to neural population data. Buesing et al. 2012 NIPS]
% - Nuclear norm penalized rate estimation 'NucNormMin' [Robust learning of low-dimensional dynamics from large neural ensembles. Pfau et al. NIPS 2013]
% - Exponential family 'ExpFamPCA'

params = GCLDSInitialize(seq,xDim, K, 'NucNormMin',params); 
%%
params.model.notes.g_ridge_lam = 1;

%by default use Variational Inference (which can be slow), 
%also recommend running Laplace-EM first and then switch to VBEM to refine

%use Variational Bayes
%params.model.inferenceHandle = @GCLDSVariationalInference;
%use Laplace inference
params.model.inferenceHandle = @GCLDSLaplaceInference;  % comment out for using variational infernce

params.opts.algorithmic.EMIterationbortDecresingVarBound = 1;
params.opts.algorithmic.EMIterations.maxIter     = maxIter;						% setting max no of EM iterations
params.opts.algorithmic.EMIterations.maxCPUTime  = 200;		% setting max CPU time for EM to 600s

tic; [params seq varBound EStepTimes MStepTimes] = GCLDSPopSpikeEM(params,seq); toc;    % do EM iterations

fprintf('Subspace angle between true and GCLDS loading matrices: %d\n',subspace(tp.model.C,params.model.C))
fprintf('Subspace angle between true and initial loading matrices: %d\n',subspace(tp.model.C,params.modelInit.C))

%%

%%%% compare ground truth and estimated model

figure;
plot(varBound)
title('expected log-likelihood as function of EM iterations')

hf = figure(); hold on
plotMatrixSpectrum(tp.model.A,'figh',hf,'col','k');
plotMatrixSpectrum(params.model.A,'figh',hf,'col','r');
title('true (black) and estimated (red) eigenvalues of dynamics matrix')

tp.model.Pi     = dlyap(tp.model.A,tp.model.Q);
params.model.Pi = dlyap(params.model.A,params.model.Q);

figure
plot(vec(tp.model.C*tp.model.Pi*tp.model.C'),vec(params.model.C*params.model.Pi*params.model.C'),'xr')
title('true vs estimated elements of the stationary rate covariance matrix')

figure
plot(vec(tp.model.C*tp.model.A*tp.model.Pi*tp.model.C'),vec(params.model.C*params.model.A*params.model.Pi*params.model.C'),'xr')
title('true vs estimated elements of one time step delayed cross-covariance matrix')

figure
l1 = plot(0:K, [zeros(1, yDim);params.model.g'], '+--');
hold on;
l2 = plot(0:length(g), [0,g], 'LineWidth',2);
title('true vs estimated g() function')
legend([l2, l1(1)], 'true', 'estimation (for each neuron)');
