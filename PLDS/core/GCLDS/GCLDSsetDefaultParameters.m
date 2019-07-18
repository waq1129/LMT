function params = GCLDSsetDefaultParameters(params,xDim,yDim,K)
%
% params = GCLDSsetDefaultParameters(params,xDim,yDim)
%
%
% Lars Buesing, Jakob H Macke, Yuanjun Gao
%
% xDim: dim for latent space
% yDim: number of neurons
% K:    maximum number of spikes (max of support of g)


%%%%%%%%%%% set standard parameters %%%%%%%%%%%%%%%%%%%%
% these standard settings make sense for data binned at 10ms with average rates of roughly 10Hz
params = touchField(params,'model');
params.model = touchField(params.model,'A',0.9*eye(xDim));    % dynamics matrix A                              
params.model = touchField(params.model,'Q',(1-0.9.^2)*eye(xDim)); %innovation covariance Q
params.model = touchField(params.model,'Q0',eye(xDim)); %initial state covariance Q0
params.model = touchField(params.model,'x0',zeros(xDim,1)); %initial mean x0
params.model = touchField(params.model,'C',randn(yDim,xDim)./sqrt(xDim)); %couplings from latent to observed C
params.model = touchField(params.model,'d',zeros(yDim,1)); %needed for LDS sample
%%mean-controlling offset d for each neuron, not needed because we have g
params.model = touchField(params.model,'B',zeros(xDim,0));
params.model = touchField(params.model,'g',repmat((1:K)*(-2),yDim,1));

params.model = touchField(params.model,'notes');
params.model.notes = touchField(params.model.notes,'learnx0', true);
params.model.notes = touchField(params.model.notes,'learnQ0', true);
params.model.notes = touchField(params.model.notes,'learnA',  true);
params.model.notes = touchField(params.model.notes,'learnR',  false);
params.model.notes = touchField(params.model.notes,'useR',    false);
params.model.notes = touchField(params.model.notes,'useB',    false);
params.model.notes = touchField(params.model.notes,'useS',    false);
params.model.notes = touchField(params.model.notes,'useCMask',false);
params.model.notes = touchField(params.model.notes,'gStatus',2);
params.model.notes = touchField(params.model.notes,'g_ridge_lam',1); %ridge penalty coefficient for second difference of g


%%%%%%%%%%% set standard observation model handles for variational inference %%%%%%%%%%%%%%%%%%%%
%note: these functions all have to be consistent with each other, i.e.
%the likeHandle, dualHandle, domainHandle, baseMeasureHandle, MstepHandle all 
%have to be corresponding to the same likelihood-model and inference
%procedure, otherwise funny things will happen.
%at the moment, only Poisson model with exponential nonlinarity is
%implemented

params.model = touchField(params.model,'likeHandle',       @ExpGPoissonHandle); %use exponential Poisson likelihood
params.model = touchField(params.model,'dualHandle',       @ExpGPoissonDualHandle); %and its dual
params.model = touchField(params.model,'domainHandle',     @ExpGPoissonDomain); %specify the domain of addmissable parameters
%params.model = touchField(params.model,'baseMeasureHandle',@PoissonBaseMeasure); %base measure, i.e. constant part which does not need to be evaluated at each step
params.model = touchField(params.model,'inferenceHandle',  @GCLDSVariationalInference); % function that does the actual inference
params.model = touchField(params.model,'MStepHandle',      @GCLDSMStep); %handle to function that does the M-step
params.model = touchField(params.model,'ParamPenalizerHandle',@PLDSemptyParamPenalizerHandle); %no penalty at all


%%%%%%%%%%% set standard algorithmic parameters %%%%%%%%%%%%%%%%%%%%

params = touchField(params,'opts');
params.opts = touchField(params.opts,'algorithmic');


%%%% set parameters for Variational Inference %%%%
%these parameters are handed over to the function 'minFunc' that is used
%for optimization, so see the documentation of minFunc for what the
%parameters mean and do
params.opts.algorithmic = touchField(params.opts.algorithmic,'VarInfX');
params.opts.algorithmic.VarInfX = touchField(params.opts.algorithmic.VarInfX,'minFuncOptions');

params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'display',	'none'); 
params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'maxFunEvals',50000);  
params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'MaxIter',	5000);
params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'progTol',	1e-6); % this might be too agressive, maybe 1e-9 is the better option
params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'optTol',	1e-5);
params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'Method',	'lbfgs');


%%%% set parameters for MStep of observation model %%%%%%%%
%these parameters are handed over to the function 'minFunc' that is used
%for optimization, so see the documentation of minFunc for what the
%parameters mean and do
params.opts.algorithmic = touchField(params.opts.algorithmic,'MStepObservation');
params.opts.algorithmic.MStepObservation = touchField(params.opts.algorithmic.MStepObservation,'minFuncOptions');

params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'maxFunEvals', 5000);
params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'MaxIter',	  500);
params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'Method',	  'lbfgs');
params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'progTol',     1e-9);
params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'optTol',      1e-5);
params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'display',	  'none');


%%%% set parameters for EM iterations %%%%%%%%       

params.opts.algorithmic = touchField(params.opts.algorithmic,'TransformType','0');         % transform LDS parameters after each MStep to canonical form?
params.opts.algorithmic = touchField(params.opts.algorithmic,'EMIterations');
params.opts.algorithmic.EMIterations = touchField(params.opts.algorithmic.EMIterations,'maxIter',100);			% max no of EM iterations
params.opts.algorithmic.EMIterations = touchField(params.opts.algorithmic.EMIterations,'maxCPUTime',inf);		% max CPU time for EM
params.opts.algorithmic.EMIterations = touchField(params.opts.algorithmic.EMIterations,'progTolvarBound',1e-6);     	% progress tolerance on var bound per data time bin
params.opts.algorithmic.EMIterations = touchField(params.opts.algorithmic.EMIterations,'abortDecresingVarBound',true);


%%%% set parameters for initialization methods %%%%

params.opts = touchField(params.opts,'initMethod','params');

switch params.opts.initMethod

   case {'params', 'params_PLDS'}
   	% do nothing
	
   case 'PLDSID'
    %use Poisson-Linear-Dynamics System Identification Method, see
    %documentation of 'FitPLDSParamsSSID' for details
   	params.opts.algorithmic = touchField(params.opts.algorithmic,'PLDSID');
	params.opts.algorithmic.PLDSID = touchField(params.opts.algorithmic.PLDSID,'algo','SVD');
	params.opts.algorithmic.PLDSID = touchField(params.opts.algorithmic.PLDSID,'hS',xDim);
	params.opts.algorithmic.PLDSID = touchField(params.opts.algorithmic.PLDSID,'minFanoFactor',1.01);
	params.opts.algorithmic.PLDSID = touchField(params.opts.algorithmic.PLDSID,'minEig',1e-4);
	params.opts.algorithmic.PLDSID = touchField(params.opts.algorithmic.PLDSID,'useB',0);
	params.opts.algorithmic.PLDSID = touchField(params.opts.algorithmic.PLDSID,'doNonlinTransform',1);

   case 'ExpFamPCA'
    %use Exponential Family PCA, see function ExpFamPCA for details
   	params.opts.algorithmic = touchField(params.opts.algorithmic,'ExpFamPCA');
	params.opts.algorithmic.ExpFamPCA = touchField(params.opts.algorithmic.ExpFamPCA,'dt',10);				% rebinning factor, choose such that roughly E[y_{kt}] = 1 forall k,t
	params.opts.algorithmic.ExpFamPCA = touchField(params.opts.algorithmic.ExpFamPCA,'lam',1);		  		% regularization coeff for ExpFamPCA
	params.opts.algorithmic.ExpFamPCA = touchField(params.opts.algorithmic.ExpFamPCA,'options');				% options for minFunc
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'display','none');
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'MaxIter',10000);
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'maxFunEvals',50000);
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'Method','lbfgs');
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'progTol',1e-9);
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'optTol',1e-5); 


   case 'NucNormMin'
    %use Exponential Family PCA, see function MODnucnrmminWithd for details
   	params.opts.algorithmic = touchField(params.opts.algorithmic,'NucNormMin');
	params.opts.algorithmic.NucNormMin = touchField(params.opts.algorithmic.NucNormMin,'dt',10);
	params.opts.algorithmic.NucNormMin = touchField(params.opts.algorithmic.NucNormMin,'fixedxDim',true);
	params.opts.algorithmic.NucNormMin = touchField(params.opts.algorithmic.NucNormMin,'options');
        params.opts.algorithmic.NucNormMin.options = touchField(params.opts.algorithmic.NucNormMin.options,'rho',	1.3);
        params.opts.algorithmic.NucNormMin.options = touchField(params.opts.algorithmic.NucNormMin.options,'eps_abs',	1e-6);
        params.opts.algorithmic.NucNormMin.options = touchField(params.opts.algorithmic.NucNormMin.options,'eps_rel',	1e-3);
        params.opts.algorithmic.NucNormMin.options = touchField(params.opts.algorithmic.NucNormMin.options,'maxIter',  	250);
        params.opts.algorithmic.NucNormMin.options = touchField(params.opts.algorithmic.NucNormMin.options,'nlin',     	'exp');
        params.opts.algorithmic.NucNormMin.options = touchField(params.opts.algorithmic.NucNormMin.options,'lambda',	0.03);
        params.opts.algorithmic.NucNormMin.options = touchField(params.opts.algorithmic.NucNormMin.options,'verbose',	0);

   otherwise
	
	warning('Unknown PLDS initialization method, cannot set parameters')

end
