function Mu = getPriorMeanLDS(params,T,varargin)
%
%
%

seq  = [];
A    = params.model.A;
x0   = params.model.x0;
xAdd = [];

assignopts(who,varargin);

xDim = size(params.model.A,1);

Mu = zeros(xDim,T);
Mu(:,1) = x0;

if params.model.notes.useB
  if ~isempty(seq)
    Mu(:,1) = Mu(:,1)+params.model.B*seq.u(:,1);
  else
    error('params.model.notes.useB == true   but no seq given!')
  end
end

if ~isempty(xAdd)
  Mu(:,1) = Mu(:,1)+xAdd(:,1);
end


for t=2:T
  Mu(:,t) = A*Mu(:,t-1);

  if ~isempty(seq) && params.model.notes.useB
    Mu(:,t) = Mu(:,t)+params.model.B*seq.u(:,t);
  end
  if ~isempty(xAdd)
    Mu(:,t) = Mu(:,t)+xAdd(:,t);
  end
  
end