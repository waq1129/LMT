function params = LDSsetDefaultParameters(params,xDim,yDim)
%
% params = LDSsetDefaultParameters(params,xDim,yDim)
%
%
% Lars Buesing, Jakob H Macke
%


%%%%%%%%%%% set standard parameters %%%%%%%%%%%%%%%%%%%%

params = touchField(params,'model');
params.model = touchField(params.model,'A',0.9*eye(xDim));        % dynamics matrix A                              
params.model = touchField(params.model,'Q',(1-0.9.^2)*eye(xDim)); %innovation covariance Q
params.model = touchField(params.model,'Q0',eye(xDim));           % initial state covariance Q0
params.model = touchField(params.model,'x0',zeros(xDim,1));       %initial mean x0
params.model = touchField(params.model,'C',randn(yDim,xDim)./sqrt(xDim)); %couplings from latent to observed C
params.model = touchField(params.model,'d',zeros(yDim,1)-2.0);    %mean-controlling offset d for each neuron
params.model = touchField(params.model,'B',zeros(xDim,0));
params.model = touchField(params.model,'R',0.1*eye(yDim));        % private noise covariance matrix

params.model = touchField(params.model,'notes');
params.model.notes = touchField(params.model.notes,'learnx0', true);
params.model.notes = touchField(params.model.notes,'learnQ0', true);
params.model.notes = touchField(params.model.notes,'learnA',  true);
params.model.notes = touchField(params.model.notes,'learnR',  true);
params.model.notes = touchField(params.model.notes,'diagR',   true);  % learn diagonal private variances?
params.model.notes = touchField(params.model.notes,'useR',    true);
params.model.notes = touchField(params.model.notes,'useB',    false);
params.model.notes = touchField(params.model.notes,'useS',    false);
params.model.notes = touchField(params.model.notes,'useCMask',false);


%%%%%%%%%%% set standard observation model handles for variational inference %%%%%%%%%%%%%%%%%%%%

params.model = touchField(params.model,'inferenceHandle',     @LDSInference);
params.model = touchField(params.model,'MStepHandle',         @LDSMStep); 
params.model = touchField(params.model,'ParamPenalizerHandle',@LDSemptyParamPenalizerHandle);


%%%%%%%%%%% set standard algorithmic parameters %%%%%%%%%%%%%%%%%%%%

params = touchField(params,'opts');
params.opts = touchField(params.opts,'algorithmic');


%%%% set parameters for MStep of observation model %%%%%%%%
%these parameters are handed over to the function 'minFunc' that is used
%for optimization, so see the documentation of minFunc for what the parameters mean and do
params.opts.algorithmic = touchField(params.opts.algorithmic,'MStepObservation');


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

   case 'params'
   	% do nothing
	
   case 'SSID'
    %Subspace System Identification Method
   	params.opts.algorithmic = touchField(params.opts.algorithmic,'SSID');
	params.opts.algorithmic.SSID = touchField(params.opts.algorithmic.SSID,'algo','SVD');
	params.opts.algorithmic.SSID = touchField(params.opts.algorithmic.SSID,'hS',xDim);
	params.opts.algorithmic.SSID = touchField(params.opts.algorithmic.SSID,'useB',0);
	params.opts.algorithmic.SSID = touchField(params.opts.algorithmic.SSID,'doNonlinTransform',0);

   otherwise
    warning('Unknown LDS initialization method, cannot set parameters')

end
