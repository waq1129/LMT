% PLDS toolbox example
%
% Lars Buesing, Jakob H Macke, 2014
%


clear all
close all


% set parameters for the ground truth PLDS model
 
xDim    = 5;												% latent dimensiom
yDim    = 100;											    	% observed dimension = no of neurons
T       = 100;												% no of time bins per trial; here a time step is approx 10ms 
Trials  = 25;		    										% no trials
maxIter = 100;		    										% max no of EM iterations for fitting the model


%%%% ground truth model

trueparams = PLDSgenerateExample('xDim',xDim,'yDim',yDim,'doff',-2.0);                                  % create ground truth model parameters
seqOrig    = PLDSsample(trueparams,T,Trials);								% sample from the model
tp         = trueparams;										

% print out some statistics of the artificial data
fprintf('Max spike count:    %i \n', max(vec([seqOrig.y])))
fprintf('Mean spike count:   %d \n', mean(vec([seqOrig.y])))
fprintf('Freq non-zero bin:  %d \n', mean(vec([seqOrig.y])>0.5))


%%%% fitting a model to artificial data

% the input data is in the structure seq, where seq(tr) is the data
% from trial tr
% 
% this struct has the fields:
% - seq.y   of dimension N x T for a spike raster from N neurons recorded for T time-steps
% - seq.T   2nd dimension of seq.y
% - [optional] seq.x   true value of latent variables used to generate seq.y
% - [optional] seq.yr  the rate of the Poisson process used to generate seq.y
 
seq    = seqOrig;
params = [];

% initialize parameters, options are:
% - Poisson SSID 'PLDSID', see [Spectral learning of linear dynamics from generalised-linear observations with application to neural population data. Buesing et al. 2012 NIPS]
% - Nuclear norm penalized rate estimation 'NucNormMin' [Robust learning of low-dimensional dynamics from large neural ensembles. Pfau et al. NIPS 2013]
% - Exponential family 'ExpFamPCA'
params = PLDSInitialize(seq,xDim,'NucNormMin',params);

params.model.inferenceHandle = @PLDSLaplaceInference;                                                   % comment out for using variational infernce
params.opts.algorithmic.EMIterations.maxIter     = maxIter;						% setting max no of EM iterations
params.opts.algorithmic.EMIterations.maxCPUTime  = 600;							% setting max CPU time for EM to 600s
tic; [params seq varBound EStepTimes MStepTimes] = PopSpikeEM(params,seq); toc;                         % do EM iterations
fprintf('Subspace angle between true and estimated loading matrices: %d\n',subspace(tp.model.C,params.model.C))


%%%% compare ground thruth and estimated model

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
plot(tp.model.d,params.model.d,'rx');
title('true vs estimated bias parameters')

