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
% Y Gao, L Buesing 2015


Trials          = numel(seq); 
maxIter         = params.opts.algorithmic.EMIterations.maxIter;
progTolvarBound = params.opts.algorithmic.EMIterations.progTolvarBound;  
maxCPUTime      = params.opts.algorithmic.EMIterations.maxCPUTime;

ParamPenalizerHandle = params.model.ParamPenalizerHandle;
InferenceMethod      = params.model.inferenceHandle;
MstepMethod          = params.model.MStepHandle;


EStepTimes      = nan(maxIter,1);
MStepTimes      = nan(maxIter,1);
varBound        = nan(maxIter,1);
PREVparams      = params;			     % params for backtracking!
NOWparams       = params;
varBoundMax     = -inf;


disp(['Starting PopSpikeEM using InferenceMethod  >>' char(InferenceMethod) '<<    and MStepMethod  >>' char(MstepMethod) '<<'])
fprintf('----------------------------------------------------------------------------------------------------------------------------\n')


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
      [seq, varBound(ii)] = InferenceMethod(NOWparams,seq);            %For variational method, varBound for each trials is saved in seq.posterior... ?
    % catch
    %  disp('Error in inference, aborting EM iterations')
    %  break
    %end
    infTimeEnd     = cputime;
    EStepTimes(ii) = infTimeEnd-infTimeBegin;

        % add regularizer costs to varBound !!!
    varBound(ii) = varBound(ii) - ParamPenalizerHandle(NOWparams);    
    %ridge penalty
    if isfield(params.model.notes, 'g_ridge_lam'),
        g_ridge_lam = max(0, params.model.notes.g_ridge_lam);
    else
        g_ridge_lam = 0;
    end
    g = params.model.g;
    yDim = size(g, 1);
    g_full = [zeros(yDim,1), g];
    g_diff = diff(g_full, 2, 2);
    g_pen = g_ridge_lam * sum(g_diff(:).^2);
    varBound(ii) = varBound(ii) - g_pen;

    % check termination criteria
    if params.opts.algorithmic.EMIterations.abortDecresingVarBound && (varBound(ii)<varBoundMax)    % check if varBound is increasing!
       NOWparams = PREVparams;	   % parameter backtracking
       %fprintf('\n ');
       warning('Variational lower bound is decreasing, aborting EM & backtracking\n');
       break;
    end

    if params.opts.algorithmic.EMIterations.abortDecresingVarBound && ((abs(varBound(ii)-varBoundMax)/Tall)<progTolvarBound)
       fprintf('Reached progTolvarBound for EM, aborting\n')
       break
    end	     

    varBoundMax = varBound(ii);
    PREVparams  = NOWparams;

    %%%%%%% M-step

    mstepTimeBegin = cputime;
    [NOWparams, seq] = MstepMethod(NOWparams,seq);
    mstepTimeEnd = cputime;
    MStepTimes(ii) = mstepTimeEnd-mstepTimeBegin;	  
    %{  
    catch
    NOWparams = PREVparams;     % parameter backtracking
    fprintf('\n ');
    warning('Aborting EM & backtracking');
    disp('Error in PopSpikeEM')
    break
  end
  %}
    fprintf('Iteration: %-3i     Elapsed time (EStep): %.2e     Elapsed time (MStep): %.2e     Variational Bound: %.6e\n',ii,EStepTimes(ii),MStepTimes(ii),varBound(ii))
    if (cputime-EMbeginTime)>maxCPUTime
       fprintf('Reached maxCPUTime for EM, aborting\n')
       break
    end
    
end

if ii == maxIter && (cputime-EMbeginTime)<=maxCPUTime,
    fprintf('Reached maxIter for EM\n');
end

NOWparams.opts = rmfield(NOWparams.opts,'EMiter');

fprintf('----------------------------------------------------------------------------------------------------------------------------\n')
disp('EM iterations done')