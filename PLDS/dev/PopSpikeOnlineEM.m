function [NOWparams seq varBound EStepTimes MStepTimes] = PopSpikeOnlineEM(params,seq)
%
% [NOWparams seq varBound EStepTimes MStepTimes] = PopSpikeEM(params,seq)
%
% Expectation maximization algorithm for learning parameters of population model with spikes
%
% input:
% params:       struct,  see PopSikeEngine.m for a definition and description
% seq:          struct with multiple elements, see PopSikeEngine.m for a defintion and description
%
% output: 
% NOWparams:    struct, same as input-params but with updated and added fields
% seq:          struct, same as input-struct but with added field 'posterior'
% varBound:     vector, variational bound (or other cost function) for each  iteration of EM
% EStepTimes, MStepTimes: vector, cpu-time taken by each iteration
%
% (c) L Buesing 01/2014


%!!! additional online EM parameters, put this into PLDSsetDefaultParams
%!!!
eta0     = 1;
alphaOEM = 0.5;     % keeping rate
miniBatchOEM = 10;        % size of mini batches
params.opts.algorithmic.EMIterations.abortDecresingVarBound = false;
%!!!
%!!!


Trials          = numel(seq); 
maxIter         = params.opts.algorithmic.EMIterations.maxIter;
progTolvarBound = params.opts.algorithmic.EMIterations.progTolvarBound;  
maxCPUTime      = params.opts.algorithmic.EMIterations.maxCPUTime;

ParamPenalizerHandle = params.model.ParamPenalizerHandle;
InferenceMethod      = params.model.inferenceHandle;
MstepMethod          = params.model.MStepHandle;


EStepTimes      = nan(maxIter,1);
MStepTimes      = nan(maxIter+1,1);
varBound        = nan(maxIter,1);
PREVparams      = params;			     % params for backtracking!
NOWparams       = params;
varBoundMax     = -inf;


disp(['Starting PopSpikeEM using InferenceMethod  >>' char(InferenceMethod) '<<    and MStepMethod  >>' char(MstepMethod) '<<'])
disp('----------------------------------------------------------------------------------------------------------------------------')


Tall        = sum([seq.T]);
EMbeginTime = cputime;

%!!!
varBoundTr = zeros(Trials,1);
seq = InferenceMethod(NOWparams,seq);
for tr=1:Trials
  varBoundTr(tr) = seq(tr).posterior.varBound;
end


%%%%%%%%%%% outer EM loop
for ii=1:maxIter

    %%%%%%% E-step: inference

    % !!! select random minibatch
    randIdx = randperm(Trials);
    seqMini = seq(randIdx(1:miniBatchOEM));
    
    % do inference
    infTimeBegin = cputime;
    NOWparams.opts.EMiter = ii;
    %try 
      seqMini = InferenceMethod(NOWparams,seqMini);            %For variational method, varBound for each trials is saved in seq.posterior... ?
    % catch
    %  disp('Error in inference, aborting EM iterations')
    %  break
    %end
    infTimeEnd     = cputime;
    EStepTimes(ii) = infTimeEnd-infTimeBegin;

    % add regularizer costs to varBound
    for tr=1:miniBatchOEM
      varBoundTr(randIdx(tr)) = seqMini(tr).posterior.varBound;
    end
    varBound(ii) = sum(varBoundTr);
    %[~, varBound(ii)] = InferenceMethod(NOWparams,seq);%sum(varBoundTr);
    varBound(ii) = varBound(ii) - ParamPenalizerHandle(NOWparams);

    fprintf('\rIteration: %i     Elapsed time (EStep): %d     Elapsed time (MStep): %d     Variational Bound: %d',ii,EStepTimes(ii),MStepTimes(ii),varBound(ii))

    % check termination criteria
    if params.opts.algorithmic.EMIterations.abortDecresingVarBound && (varBound(ii)<varBoundMax)    % check if varBound is increasing!
       NOWparams = PREVparams;	   % parameter backtracking
       fprintf('\n ');
       warning('Variational lower bound is decreasing, aborting EM & backtracking');
       break;
    end

    if params.opts.algorithmic.EMIterations.abortDecresingVarBound && ((abs(varBound(ii)-varBoundMax)/Tall)<progTolvarBound)
       fprintf('\nReached progTolvarBound for EM, aborting')
       break
    end	     

    if (cputime-EMbeginTime)>maxCPUTime
       fprintf('\nReached maxCPUTime for EM, aborting')
       break
    end

    varBoundMax = varBound(ii);
    PREVparams  = NOWparams;


    %%%%%%% M-step

    etaOEM = eta0*(2+ii)^(-alphaOEM);

    mstepTimeBegin = cputime;
    Miniparams = MstepMethod(NOWparams,seqMini);
    pFields = {'A','Q','Q0','x0','C','d'};
    for jj=1:numel(pFields)
      newp = (1-etaOEM)*getfield(NOWparams.model,pFields{jj})+etaOEM*getfield(Miniparams.model,pFields{jj});
      NOWparams.model = setfield(NOWparams.model,pFields{jj},newp);
    end
    mstepTimeEnd = cputime;
    MStepTimes(ii+1) = mstepTimeEnd-mstepTimeBegin;	      

end

NOWparams.opts = rmfield(NOWparams.opts,'EMiter');

fprintf('\n----------------------------------------------------------------------------------------------------------------------------\n')
disp('EM iterations done')