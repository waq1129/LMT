function params = PLDSLinkFuncInitialize(seq,xDim,initMethod,params)
%
% function params = PLDSLinkFuncInitialize(seq,xDim,initMethod,params)
%
% Inititalize parameters of Population-LDS model with different methods. 
% This method allows for arbitrary link function
%
%input: 
% seq:      standard data struct
% xdim:     desired latent dimensionality
% initMethods:
%
% - params	
% - ExpFamPCA	
% - NucNormMin
%
% (c) L Buesing 2014


yDim       = size(seq(1).y,1);                                                                           
Trials     = numel(seq);
params.opts.initMethod = initMethod;
params     = PLDSLinkFuncsetDefaultParameters(params,xDim,yDim);


switch initMethod

 case 'params'
  % do nothing
  disp('Initializing PLDS parameters with given parameters')
  
 case 'ExpFamPCA'     
  % this replaces the FA initializer from the previous verions...
  disp('Initializing PLDS parameters using exponential family PCA with custom link function')
  
  Y  = [seq.y];
  if params.model.notes.useS; Yext = [seq.s]; else; Yext = 0;end
  [Cpca, Xpca, dpca] = ExpFamPCALinkFunc(Y,xDim,'Cinit',[],'Xinit',[],'dinit',[],'lam',params.opts.algorithmic.ExpFamPCA.lam,'options',params.opts.algorithmic.ExpFamPCA.options,'s',Yext,'linkFunc',params.model.linkFunc,'dlinkFunc',params.model.dlinkFunc);     
  params.model.Xpca = Xpca;
  params.model.C = Cpca;
  params.model.d = dpca;
  
  if params.model.notes.useB; u = [seq.u];else;u = [];end
  params.model = LDSObservedEstimation(Xpca,params.model,1,u);
  
 case 'NucNormMin'
  disp('Initializing PLDS parameters using Nuclear Norm Minimization')
  Y  = [seq.y];
  options = params.opts.algorithmic.NucNormMin.options;
  options.lambda = options.lambda*sqrt(size(Y,1)*size(Y,2));
  if params.model.notes.useS; Yext = [seq.s];
  else; Yext = [];end

  [Y,Xu,Xs,Xv,d] = MODnucnrmminWithd( Y, options , 'Yext', Yext,'f',params.model.linkFunc,'df',params.model.dlinkFunc,'d2f',params.model.d2linkFunc);
  params.model.d = d;

  if ~params.opts.algorithmic.NucNormMin.fixedxDim
    disp('Variable dimension; still to implement!')
  else
    params.model.C = Xu(:,1:xDim)*Xs(1:xDim,1:xDim);
    if params.model.notes.useB; u = [seq.u];else;u = [];end
    params.model = LDSObservedEstimation(Xv(:,1:xDim)',params.model,1,u);
    params.model.Xpca = Xv(:,1:xDim)';
    params.model.Xs   = diag(Xs(1:xDim,1:xDim));
  end

 otherwise
  warning('Unknown PLDS initialization method')
  
end

if params.model.notes.useB && (numel(params.model.B)<1)
  params.model.B = zeros(xDim,size(seq(1).u,1));
end

params = LDSTransformParams(params,'TransformType',params.opts.algorithmic.TransformType);	% clean up parameters
params.modelInit = params.model;
