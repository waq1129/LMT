error('not implemented yet')
%parameter conventions in here are out of date

%interface for using the code
%function [dat,params]=PopSpikeEngine(dat,params,opts)

%dat: data put in, same format as in PPLDS_packaged
%params: are learned by system
%opts: optional arguments
%opts.model: model options
%opts.prior: options for prior
%opts.algo: algorithm options

%
%dat.y spike trains
%dat.s stimulus/history features
%dat.u stimulus/history features
%dat.h shared stimulus

%dat.posterior.xsm posterior mean over latent
%dat.posterior.Vsm posterior covariance over latent, instantaneous
%dat.posterior.Vsm(:,:,20) is posterior covariance over x(:,20)
%dat.posterior.VVSM time-lag 1 covariance, i.e. covarariance between x(:,20) and x(:,21)

%params

%params.initparams initial guess (if one gives one)
%params.current current/final estimate of parameters
%params.runinfo stuff that is not parameters, "log file"


%calls PopSpikeInitialize
%calls PopSpikeEM or PopSpikeOnlineEM


%model:
%y_t= Cx_t +sum(D.*s_t,2) +Eu_t+ nu_t (for Gaussian)
%x_t=Ax_{t-1}+Bh_t+ epsilon_t

%s_t: stimulus not shared across neurons, i.e. Ds_t '=' sum(D.*s_t,2)
%u_t: shared stimulus across neurons
%h_t: stimulus into latent





