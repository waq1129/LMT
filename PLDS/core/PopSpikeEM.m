function [NOWparams seq varBound EStepTimes MStepTimes] = PopSpikeEM(params,seq)
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

%%%%%%%%%%% outer EM loop
for ii=1:maxIter
    %try
    NOWparams.state.EMiter = ii;
    
    %%%%%%% E-step: inference
    
    % do inference
    infTimeBegin = cputime;
    NOWparams.opts.EMiter = ii;
    %try
    [seq varBound(ii)] = InferenceMethod(NOWparams,seq);            %For variational method, varBound for each trials is saved in seq.posterior... ?
    % catch
    %  disp('Error in inference, aborting EM iterations')
    %  break
    %end
    infTimeEnd     = cputime;
    EStepTimes(ii) = infTimeEnd-infTimeBegin;
    
    % add regularizer costs to varBound !!!
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
    
    mstepTimeBegin = cputime;
    [NOWparams seq] = MstepMethod(NOWparams,seq);
    %NOWparams = PLDSMStep(NOWparams,seq);
    mstepTimeEnd = cputime;
    MStepTimes(ii+1) = mstepTimeEnd-mstepTimeBegin;
    %{
catch
    NOWparams = PREVparams;     % parameter backtracking
    fprintf('\n ');
    warning('Aborting EM & backtracking');
    disp('Error in PopSpikeEM')
    break
  end
    %}
end

NOWparams.opts = rmfield(NOWparams.opts,'EMiter');

fprintf('\n----------------------------------------------------------------------------------------------------------------------------\n')
disp('EM iterations done')