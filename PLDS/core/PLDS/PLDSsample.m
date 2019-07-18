function seq = PLDSsample(params,T,Trials,varargin)
%
% seq = PLDSsample(params,T,Trials)
%
% sample from a PLDS model with exponential nonlinearity (or
% user defined link function); uses LDSsample
%
% (c) L Buesing 2014
%



yMax = inf;
assignopts(who,varargin);

if isfield(params.model,'linkFunc')
  linkFunc = params.model.linkFunc;
  %disp('Using non-exp link function for sampling')
else
  linkFunc = @exp;
end

seq = LDSsample(params,T,Trials,varargin{:});


for tr=1:Trials
  seq(tr).yr   = real(seq(tr).y);
  seq(tr).y    = poissrnd(linkFunc(seq(tr).yr));
  seq(tr).y(:) = min(yMax,seq(tr).y(:));
end

