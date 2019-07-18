function params = LDSInitialize(seq,xDim,initMethod,params)
%
% function params = LDSInitialize(params,seq) 
%
% inititalize parameters of LDS model
% the moment, focusses on exponential link function and Poisson
% observations.
%
%input: 
% seq:      standard data struct
% xdim:     desired latent dimensionality
%
% initMethods:
%
% - params											% just initialize minimal undefiened fields with standard values
% - SSID											% moment conversion + Ho-Kalman SSID
% - PCA
%
% (c) L Buesing 2014


yDim       = size(seq(1).y,1);                                                                           
Trials     = numel(seq);
params.opts.initMethod = initMethod;
params     = LDSsetDefaultParameters(params,xDim,yDim);					% set standard parameter values


switch initMethod

   case 'params'
   	% do nothing
	disp('Initializing PLDS parameters with given parameters')

       
   case 'SSID'
   	% !!! debugg SSID stuff separately & change to params.model convention
	disp('Initializing LDS parameters using SSID')
	if params.model.notes.useB
	  warning('SSID initialization with external input: not implemented yet!!!')
	end
	SSIDoptions  = struct2arglist(params.opts.algorithmic.SSID);
	params.model = FitPLDSParamsSSID(seq,xDim,'params',params.model,SSIDoptions{:});


   otherwise
	warning('Unknown PLDS initialization method')

end

if params.model.notes.useB && (numel(params.model.B)<1)
  params.model.B = zeros(xDim,size(seq(1).u,1));
end

params = LDSTransformParams(params,'TransformType',params.opts.algorithmic.TransformType);	% clean up parameters
params.modelInit = params.model;
