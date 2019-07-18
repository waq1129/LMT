% PLDS toolbox example
%
% Lars Buesing, Jakob H Macke, 2014
%
% clear all
function [x11,params] = run_plds(yy,nf)
% cd pop_spike_dyn
addpath(genpath(pwd)); warning off

% set parameters for the ground truth PLDS model
xDim    = nf;												% latent dimensiom
yDim    = size(yy,2);											    	% observed dimension = no of neurons
T       = size(yy,1);												% no of time bins per trial; here a time step is approx 10ms
Trials  = 1;		    										% no trials
maxIter = 100;		    										% max no of EM iterations for fitting the model
xx = zeros(T,xDim);

%%%% ground truth model

trueparams = PLDSgenerateExample('xDim',xDim,'yDim',yDim,'doff',-2.0);                                  % create ground truth model parameters
seqOrig    = PLDSsample(trueparams,T,Trials);								% sample from the model
seqOrig.x = xx';
seqOrig.y = sqrt(yy)';
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
params.model.inferenceHandle = @PLDSLaplaceInference;                           % comment out for using variational infernce
params.opts.algorithmic.EMIterations.maxIter     = maxIter;						% setting max no of EM iterations
params.opts.algorithmic.EMIterations.maxCPUTime  = 600;							% setting max CPU time for EM to 600s
tic; [params seq varBound EStepTimes MStepTimes] = PopSpikeEM(params,seq); toc;                         % do EM iterations
fprintf('Subspace angle between true and estimated loading matrices: %d\n',subspace(tp.model.C,params.model.C))

%%%% compare ground thruth and estimated model
post = seq.posterior;
x0 = seq.x;
x1 = post.xsm;
% x11 = norm_xtrue(x1',x0')';
x11 = x1;

% subplot(221),plot([x0(1,:)' x11(1,:)'])
% subplot(222),plot([x0(2,:)' x11(2,:)'])
% 
% cc = norm_xtrue(params.model.C,tp.model.C);
% subplot(223),plot([tp.model.C(:,1) cc(:,1)])
% subplot(224),plot([tp.model.C(:,2) cc(:,2)])
% %pause
% save(['sps_ori' num2str(iori) '_trial' num2str(jjrpt) '_plds'],'x11')





