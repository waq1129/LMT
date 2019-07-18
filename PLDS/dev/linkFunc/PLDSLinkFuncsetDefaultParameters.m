function params = PLDSLinkFuncsetDefaultParameters(params,xDim,yDim)
%
% function params = PLDSLinkFuncsetDefaultParameters(params,xDim,yDim)
%
% Lars Buesing, Jakob H Macke
%



%%%%%%%%%%%%%%%%%%%%% set standard link function %%%%%%%%%%%%%%%%%%%%

params = touchField(params,'model');
params.model = touchField(params.model,'linkFunc',         @(x) log(1+exp(x)));%@stableLogExp);% log(1+exp(x)));
params.model = touchField(params.model,'dlinkFunc',        @(x) 1./(1+exp(-x)));
params.model = touchField(params.model,'d2linkFunc',       @(x) exp(-x)./(1+exp(-x)).^2);%1./(exp(-x/2)+exp(x/2)).^2);
params.model = touchField(params.model,'stablelinkFunc',   @stableLogExp);
params.model = touchField(params.model,'stableloglinkFunc',@stableLogLogExp);


%%%%%%%%%%%%%%%%%%%% generate function tables for expected log-likelihood %%%%%%%%%%%%%% 

if ~isfield(params.model,'ftabs')
  disp('Pre-computing Eq[log p(y|x)] tables...')
  [params.model.ftabs.linkFunctab,params.model.ftabs.loglinkFunctab,params.model.ftabs.Mus,params.model.ftabs.Stds] = makeEqLinkFunctables(params.model.stablelinkFunc,params.model.stableloglinkFunc);
  disp('done')
end


%%%%%%%%%%% set standard observation model handles for variational inference %%%%%%%%%%%%%%%%%%%%

params.model = touchField(params.model,'likeHandle',          []); 
params.model = touchField(params.model,'dualHandle',          []);
params.model = touchField(params.model,'domainHandle',        []); 
params.model = touchField(params.model,'baseMeasureHandle',   []); 
params.model = touchField(params.model,'inferenceHandle',     @PLDSLinkFuncLaplaceInference); % function that does the actual inference
params.model = touchField(params.model,'MStepHandle',         @PLDSLinkFuncMStep); %handle to function that does the M-step
params.model = touchField(params.model,'ParamPenalizerHandle',@PLDSemptyParamPenalizerHandle);


%%%%%%%%%%% set standard algorithmic parameters %%%%%%%%%%%%%%%%%%%%

params = touchField(params,'opts');
params.opts = touchField(params.opts,'algorithmic');


%%%% set parameters for Variational Inference %%%%

params.opts.algorithmic = touchField(params.opts.algorithmic,'VarInfX');
params.opts.algorithmic.VarInfX = touchField(params.opts.algorithmic.VarInfX,'minFuncOptions');

params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'display',	 []); 
params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'maxFunEvals',[]);  
params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'MaxIter',	 1000);
params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'progTol',	 1e-10); % this might be too agressive, maybe 1e-9 is the better option
params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'optTol',	 []);
params.opts.algorithmic.VarInfX.minFuncOptions = touchField(params.opts.algorithmic.VarInfX.minFuncOptions,'Method',	 []);


%%%% set parameters for MStep of observation model %%%%%%%%

params.opts.algorithmic = touchField(params.opts.algorithmic,'MStep');
params.opts.algorithmic.MStep = touchField(params.opts.algorithmic.MStep,'subPostMean', true);
params.opts.algorithmic.MStep = touchField(params.opts.algorithmic.MStep,'smoothPostMean',3);


params.opts.algorithmic = touchField(params.opts.algorithmic,'MStepObservation');
params.opts.algorithmic.MStepObservation = touchField(params.opts.algorithmic.MStepObservation,'minFuncOptions');

params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'maxFunEvals', []);
params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'MaxIter',	    100);
params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'Method',	    []);
params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'progTol',     1e-10);
params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'optTol',      []);
params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'display',	    []);
params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'gamma0',      0.5);
params.opts.algorithmic.MStepObservation.minFuncOptions = touchField(params.opts.algorithmic.MStepObservation.minFuncOptions,'etaCd',       0.0);

%%%% set parameters for initialization methods %%%%

params.opts = touchField(params.opts,'initMethod','params');

switch params.opts.initMethod

   case 'params'
   	% do nothing
	
   case 'ExpFamPCA'
    %use Exponential Family PCA, see function ExpFamPCA for details
   	params.opts.algorithmic = touchField(params.opts.algorithmic,'ExpFamPCA');
	params.opts.algorithmic.ExpFamPCA = touchField(params.opts.algorithmic.ExpFamPCA,'dt',[]);				% rebinning factor, choose such that roughly E[y_{kt}] = 1 forall k,t
	params.opts.algorithmic.ExpFamPCA = touchField(params.opts.algorithmic.ExpFamPCA,'lam',6);		  		% regularization coeff for ExpFamPCA
	params.opts.algorithmic.ExpFamPCA = touchField(params.opts.algorithmic.ExpFamPCA,'options');				% options for minFunc
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'display','none');
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'MaxIter',10000);
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'maxFunEvals',50000);
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'Method','lbfgs');
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'progTol',1e-9);
	params.opts.algorithmic.ExpFamPCA.options = touchField(params.opts.algorithmic.ExpFamPCA.options,'optTol',1e-5); 


   case 'NucNormMin'
        %use Exponential Family PCA, see function MODnucnrmminWithd for
        %details
        params.opts.algorithmic = touchField(params.opts.algorithmic,'NucNormMin');
        params.opts.algorithmic.NucNormMin = touchField(params.opts.algorithmic.NucNormMin,'dt',[]);
        params.opts.algorithmic.NucNormMin = touchField(params.opts.algorithmic.NucNormMin,'fixedxDim',true);
        params.opts.algorithmic.NucNormMin = touchField(params.opts.algorithmic.NucNormMin,'options');
        params.opts.algorithmic.NucNormMin.options = touchField(params.opts.algorithmic.NucNormMin.options,'rho',       1.3);
        params.opts.algorithmic.NucNormMin.options = touchField(params.opts.algorithmic.NucNormMin.options,'eps_abs',   1e-6);
        params.opts.algorithmic.NucNormMin.options = touchField(params.opts.algorithmic.NucNormMin.options,'eps_rel',   1e-3);
        params.opts.algorithmic.NucNormMin.options = touchField(params.opts.algorithmic.NucNormMin.options,'maxIter',   250);
        params.opts.algorithmic.NucNormMin.options = touchField(params.opts.algorithmic.NucNormMin.options,'nlin',      'logexp');
        params.opts.algorithmic.NucNormMin.options = touchField(params.opts.algorithmic.NucNormMin.options,'lambda',    0.03);
        params.opts.algorithmic.NucNormMin.options = touchField(params.opts.algorithmic.NucNormMin.options,'verbose',   0);

   otherwise
	warning('Unknown PLDS initialization method, cannot set parameters')

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% fill in remaining entries with PLDS standards %%%%%%%%%%%%%%%%%%%%%%%%%%%

params = PLDSsetDefaultParameters(params,xDim,yDim);
