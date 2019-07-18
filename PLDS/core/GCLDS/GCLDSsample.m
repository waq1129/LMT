function seq = GCLDSsample(params,T,Trials,varargin)
%
% seq = GCLDSsample(params,T,Trials)
%
% sample from a GCLDS model with exponential nonlinearity (or
% user defined link function); uses LDSsample
%
% params: GCLDS parameters
% T: number of time bins
% Trials: number of trials
% varargin: other LDS variables
%
% (c) L Buesing, Y Gao 2015
%



yMax = inf;
assignopts(who,varargin);

%if isfield(params.model,'linkFunc')
%  linkFunc = params.model.linkFunc;
  %disp('Using non-exp link function for sampling')
%else
%  linkFunc = @exp;
%end

seq = LDSsample(params,T,Trials,varargin{:});


for tr=1:Trials
  seq(tr).yr   = real(seq(tr).y);
  seq(tr).y   = seq(tr).yr;
  yDim = size(seq(tr).yr);
  for i=1:yDim
    seq(tr).y(i,:)    = gcrnd(seq(tr).yr(i,:), params.model.g(i,:));
  end
  seq(tr).y(:) = min(yMax,seq(tr).y(:));
end

