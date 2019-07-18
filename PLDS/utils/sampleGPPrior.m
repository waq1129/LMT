function samples = sampleGPPrior(N,T,Bdim,varargin)
%
% samples = sampleGPPrior(N,T,Bdim,varargin)
%
%


xpos = 1:T; % sampling locations
tau  = 10; % sqaured-exp length scale
sig  = 0.01; % uncorrelated noise

assignopts(who, varargin);

sqx  = xpos.^2;
Dist = repmat(sqx,T,1)+repmat(sqx,T,1)'-2*xpos'*xpos;
K    = exp(-Dist./(tau.^2))*(1-sig.^2)+sig.^2*eye(T);
C    = chol(K)';

for n=1:N
  samples{n} = (C*randn(T,Bdim))';
end
