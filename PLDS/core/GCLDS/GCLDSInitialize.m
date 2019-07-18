function params = GCLDSInitialize(seq,xDim,K,initMethod,params)
%
% function params = GCLDSInitialize(params,seq) 
%
% inititalize parameters of GCLDS model with different methods. At
% the moment, focusses on exponential link function and Poisson
% observations.
%
%input: 
% seq:      standard data struct
% xdim:     desired latent dimensionality
% initMethods:
% - params				% just initialize minimal undefiened fields with standard values
% - params_PLDS         % use the parameters fitted by PLDS model
% - PLDSID				% moment conversion + Ho-Kalman SSID
% - ExpFamPCA			% exponential family PCA
% - NucNormMin			% nuclear norm minimization, see [Robust learning of low-dimensional dynamics from large neural ensembles David Pfau, Eftychios A. Pnevmatikakis, Liam Paninski. NIPS2013] 
% params: if initialization method 'params' or 'params_PLDS' is chosen, the params-struct that one should use
%
% (c) L Buesing, Y Gao, 2015

yDim       = size(seq(1).y,1);                                                                           
Trials     = numel(seq);
params.opts.initMethod = initMethod;
params     = GCLDSsetDefaultParameters(params,xDim,yDim,K);					% set standard parameter values


switch initMethod

   case 'params'
   	% do nothing
	disp('Initializing GCLDS parameters with given parameters')
    
    %set parameters from an existing PLDS model
   case 'params_PLDS'
       disp('Initializing GCLDS parameters using PLDS params')
       params_PLDS = params;
       params = [];
       params     = GCLDSsetDefaultParameters(params,xDim,yDim,K);
       params.model.A = params_PLDS.model.A;
       params.model.Q = params_PLDS.model.Q;
       params.model.Q0 = params_PLDS.model.Q0;
       params.model.x0 = params_PLDS.model.x0;
       params.model.C = params_PLDS.model.C;
       params.model.B = params_PLDS.model.B;
       params.model.d = params_PLDS.model.d;

       
   case 'PLDSID'
   	% !!! debugg SSID stuff separately & change to params.model convention
	disp('Initializing GCLDS parameters using PLDSID')
	if params.model.notes.useB
	  warning('SSID initialization with external input: not implemented yet!!!')
	end
	PLDSIDoptions = struct2arglist(params.opts.algorithmic.PLDSID);
	params.model = FitPLDSParamsSSID(seq,xDim,'params',params.model,PLDSIDoptions{:});


   case 'ExpFamPCA'     
   	% this replaces the FA initializer from the previous verions...
   	disp('Initializing GCLDS parameters using exponential family PCA')
	
	dt = params.opts.algorithmic.ExpFamPCA.dt;
	Y  = [seq.y];
	if params.model.notes.useS; s = [seq.s]; 
	else s=0;end
	[Cpca, Xpca, dpca] = ExpFamPCA(Y,xDim,'dt',dt,'lam',params.opts.algorithmic.ExpFamPCA.lam,'options',params.opts.algorithmic.ExpFamPCA.options,'s',s);     
	params.model.Xpca = Xpca;
	params.model.C = Cpca;
    params.model.d = dpca;

	if params.model.notes.useB; u = [seq.u];else;u = [];end
	params.model = LDSObservedEstimation(Xpca,params.model,dt,u);
	
    

   case 'NucNormMin'
	disp('Initializing GCLDS parameters using Nuclear Norm Minimization')
	
        dt = params.opts.algorithmic.NucNormMin.dt;
	seqRebin.y = [seq.y]; seqRebin = rebinRaster(seqRebin,dt);
        Y  = [seqRebin.y];
	options = params.opts.algorithmic.NucNormMin.options;
	options.lambda = options.lambda*sqrt(size(Y,1)*size(Y,2));
	if params.model.notes.useS
	  Yext = subsampleSignal([seq.s],dt);
	else
	  Yext = [];
	end
	[Y,Xu,Xs,Xv,d] = MODnucnrmminWithd( Y, options , 'Yext', Yext );
	params.model.d = d-log(dt);

	if ~params.opts.algorithmic.NucNormMin.fixedxDim
	   disp('Variable dimension; still to implement!')
	else
	   params.model.C = Xu(:,1:xDim)*Xs(1:xDim,1:xDim);
	   if params.model.notes.useB; u = [seq.u];else;u = [];end
	   params.model = LDSObservedEstimation(Xv(:,1:xDim)',params.model,dt,u);
	   params.model.Xpca = Xv(:,1:xDim)';
	   params.model.Xs   = diag(Xs(1:xDim,1:xDim));
    end
	
    

   otherwise
	warning('Unknown GCLDS initialization method')

end

if params.model.notes.useB && (numel(params.model.B)<1)
  params.model.B = zeros(xDim,size(seq(1).u,1));
end

params = LDSTransformParams(params,'TransformType',params.opts.algorithmic.TransformType);	% clean up parameters
%params.modelInit = params.model;


%K =  max(vec([seq.y]));
if params.model.notes.gStatus ~= 1,
    params.model.g = bsxfun(@times, params.model.d, 1:K); %convert d to g
elseif params.model.notes.gStatus == 1,
    params.model.g = repmat(mean(params.model.d(:)) * (1:K), yDim, 1);
end
params.model = rmfield(params.model, 'd');
%params.model.inferenceHandle = @GCLDSVariationalInference;

params.modelInit = params.model;

