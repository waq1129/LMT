function [f, df] = VariationalInferenceDualCostGCLDSBarrier(lam,rho,VarInfparams)
% 
% [f, df, varBound, m_ast, invV_ast, Vsm, VVsm, over_m, over_v] = VariationalInferenceDualCost(lam,VarInfparams)
%
% Cost function for variational inference via dual optimization for
% Gaussian LDS with exponential family observations
%
% see [M. E. Khan, A. Aravkin, M. Friedlander, and M. Seeger. Fast Dual Variational Inference for Non-Conjugate Latent Gaussian Models. In JMLR W&CP, volume 28, pages 951-959, 2013]
%
% VarInfparams.Lambda
% VarInfparams.y
% VarInfparams.mu
% VarInfparams.W
% VarInfparams.A
% VarInfparams.WlamW
% VarInfparams.g
% VarInfparams.CC
%
% OUTPUT:
% f        = dual cost
% df       = gradient of dual cost wrt lam
% varBound = variational lower bound to marignal log-likelihood log p(y)
% m_ast    = variational posterior mean xDim x T
% invV_ast = variational posterior precision
% Vsm	   = var smoothed cov Cov[x(t),x(t)|y_{1:T}]
% VVsm	   = var smoothed cov Cov[x(t+1),x(t)|y_{1:T}]
% over_m   = W*m_ast+d
% over_v   = diag(W/invV_ast*W)
%
% (c) Lars Buesing, Yuanjun Gao, 2015
%

[f, df] = VariationalInferenceDualCostGCLDS(lam,VarInfparams);
K = size(VarInfparams.g, 2);
if ~isinf(f),
    f = f - rho * sum(log(lam));
    df = df - rho * 1./lam;
    lam_mat = reshape(lam, K, []);
    lam_0 = 1-sum(lam_mat)';
    f = f - rho * sum(log(lam_0));
    df = df + rho * kron(1./lam_0, ones(K,1));
end


