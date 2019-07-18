function [NOWparams seq varBound EStepTimes MStepTimes] = LpVbIter(params,seq)
%
% [NOWparams seq varBound EStepTimes MStepTimes] = PopSpikeEM(params,seq)
%
%Expectation algorithm for learning parameters of population model with spikes
%
% input:
% params:       struct,  see PopSikeEngine.m for a definition and description
% seq:          struct with multiple elements, see PopSikeEngine.m for a defintion and description
%
% output: 
% NOWparams:    struct, same as input-params but with updated and added fields
% seq:          struct, same as input-struct but with added field
% 'posterior'
% varBound:     vector, variational bound (or other cost function) for each  iteration of EM
% EStepTimes, MStepTimes: vector, cpu-time taken by each iteration
%
% (c) L Buesing 01/2014


Trials          = numel(seq); 
maxIter         = params.opts.algorithmic.EMIterations.maxIter;
progTolvarBound = params.opts.algorithmic.EMIterations.progTolvarBound;  
maxCPUTime      = params.opts.algorithmic.EMIterations.maxCPUTime;
CostIter        = params.opts.algorithmic.EMIterations.CostIter;


ParamPenalizerHandle = params.model.ParamPenalizerHandle;
InferenceMethod      = params.model.inferenceHandle;
CostFuncMethod       = params.model.costFuncHandle;
MstepMethod          = params.model.MStepHandle;


EStepTimes      = nan(maxIter,1);
MStepTimes      = nan(maxIter+1,1);
varBound        = nan(ceil(maxIter/CostIter),1);
PREVparams      = params;			     % params for backtracking!
NOWparams       = params;
varBoundMax     = -inf;


disp(['Starting PopSpikeEM using InferenceMethod  >>' char(InferenceMethod) '<<    and MStepMethod  >>' char(MstepMethod) '<<'])
disp('----------------------------------------------------------------------------------------------------------------------------')


Tall        = sum([seq.T]);
EMbeginTime = cputime;

%%%%%%%%%%% outer EM loop
for ii=1:maxIter

    %%%%%%% E-step: inference

    % do inference
    infTimeBegin   = cputime;
    try 
    seq = InferenceMethod(NOWparams,seq);            %For variational method, varBound for each trials is saved in seq.posterior... ?
    catch
      disp('Error in inference, aborting EM iterations')
      break
    end
    infTimeEnd     = cputime;
    EStepTimes(ii) = infTimeEnd-infTimeBegin;

    if rem(ii-1,CostIter)==0
      jj  = round((ii-1)/CostIter+1);
      seqCost = CostFuncMethod(NOWparams,seq);
   
      % evaluate variational lower bound 
      varBound(jj) = 0;
      for tr=1:Trials; varBound(jj) = varBound(jj)+seqCost(tr).posterior.varBound; end;

      % add regularizer costs to varBound
      varBound(jj) = varBound(jj) - ParamPenalizerHandle(NOWparams);

      % check termination criteria
      if varBound(jj)<varBoundMax    % check if varBound is increasing!
	NOWparams = PREVparams;	   % parameter backtracking
	warning('Variational lower bound is decreasing, aborting EM & backtracking');
       break;
      end
      
      if (abs(varBound(jj)-varBoundMax)/Tall)<progTolvarBound
	fprintf('\nReached progTolvarBound for EM, aborting')
	break
      end	           

      varBoundMax = varBound(jj);
      
    end

    PREVparams = NOWparams;
    fprintf('\rIteration: %i     Elapsed time (EStep): %d       Elapsed time (MStep): %d        Variational Bound: %d',ii,EStepTimes(ii),MStepTimes(ii),varBound(jj))

    if (cputime-EMbeginTime)>maxCPUTime
       fprintf('\nReached maxCPUTime for EM, aborting')
       break
    end


    %%%%%%% M-step

    mstepTimeBegin = cputime;
    NOWparams = MstepMethod(NOWparams,seq);
    %NOWparams = PLDSMStep(NOWparams,seq);
    mstepTimeEnd   = cputime;
    MStepTimes(ii+1) = mstepTimeEnd-mstepTimeBegin;	      

end

fprintf('\n----------------------------------------------------------------------------------------------------------------------------\n')
disp('EM iterations done')