% PLDS toolbox example
% 
%
%
% Lars Buesing, Jakob H Macke, 2014
%


clear all
close all


% set parameters for the ground truth model
 
xDim    = 5;												% latent dimensiom
yDim    = 100;											    	% observed dimension = no of neurons
T       = 100;												% no of time bins per trial
Trials  = 25;		    										% no trials



%%%% ground truth model

trueparams = LDSgenerateExample('xDim',xDim,'yDim',yDim,'doff',-2.0);                                   % create ground truth model parameters
seqOrig    = LDSsample(trueparams,T,Trials);								% sample from the model


params = trueparams;
seq    = seqOrig;

% Inference

seq = LDSInference(params,seq);

% Mstep for parameters A,Q,Q0,0

params = LDSMStepLDS(params,seq);








