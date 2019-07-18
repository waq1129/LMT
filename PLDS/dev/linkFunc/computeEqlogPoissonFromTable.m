function explogPoissonLike = computeEqlogPoissonFromTable(params,Y,ZMean,ZStd)
%
% computes E_q[log p(y|eta)], where y | eta ~ Poisson(linkFunc(eta))
% and eta ~ N(ZMean,Zstd.^2*I)
%


ZMean(ZMean<min(params.model.ftabs.Mus)) = min(params.model.ftabs.Mus) +1e-3;
ZMean(ZMean>max(params.model.ftabs.Mus)) = max(params.model.ftabs.Mus) -1e-3;
ZStd(ZStd<min(params.model.ftabs.Stds))  = min(params.model.ftabs.Stds)+1e-3;
ZStd(ZStd>max(params.model.ftabs.Stds))  = max(params.model.ftabs.Stds)-1e-3;

explogPoissonLike = Y.*interp2(params.model.ftabs.Stds,params.model.ftabs.Mus,params.model.ftabs.loglinkFunctab,ZStd,ZMean);
explogPoissonLike = explogPoissonLike-interp2(params.model.ftabs.Stds,params.model.ftabs.Mus,params.model.ftabs.linkFunctab,ZStd,ZMean);
explogPoissonLike = explogPoissonLike - gammaln(Y+1);
