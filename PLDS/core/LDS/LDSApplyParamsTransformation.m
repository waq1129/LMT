function [params, seq] = LDSApplyParamsTransformation(M,params,varargin)
%
% [params seq] = LDSApplyParamsTransformation(M,params,varargin)
%
% Applies M from left and inv(M)/M' from the right
%
% to A,Q,Q0,x0,B,C
%
%
% L Buesing 2014

seq = [];

assignopts(who,varargin); 

xDim = size(params.model.A,1);

if cond(M)>1e3
   warning('Attempting LDSApplyParamsTransformation with ill-conditioned transformation')
end

params.model.C  =     params.model.C  / M;
params.model.A  = M * params.model.A  / M;
if ~iscell(params.model.Q)
  params.model.Q  = M * params.model.Q  * M';
else
  for mm=1:numel(params.model.Q)
    params.model.Q{mm}  = M * params.model.Q{mm}  * M';
  end
end

if ~iscell(params.model.Q0)
  params.model.Q0 = M * params.model.Q0 * M';
else
  for mm=1:numel(params.model.Q0)
    params.model.Q0{mm} = M * params.model.Q0{mm} * M';
  end
end
params.model.x0 = M * params.model.x0;

if isfield(params.model,'B')
  params.model.B = M*params.model.B;
end

if isfield(params.model,'Pi')
  if ~iscell(params.model.Q)
    params.model.Pi = dlyap(params.model.A,params.model.Q);
  else
    params.model.Pi = dlyap(params.model.A,params.model.Q{1});
  end
end

if ~isempty(seq)
 for tr=1:numel(seq)
   if isfield(seq,'posterior')
     seq(tr).posterior.xsm  = M*seq(tr).posterior.xsm;
     for t = 1:size(seq(tr).y,2);
       xidx = ((t-1)*xDim+1):(xDim*t);
       seq(tr).posterior.Vsm(xidx,:)  = M*seq(tr).posterior.Vsm(xidx,:)*M';
       if t>1;
	 seq(tr).posterior.VVsm(xidx-xDim,:) = M*seq(tr).posterior.VVsm(xidx-xDim,:)*M';
       end
     end
   end
   if isfield(seq,'x')
     seq(tr).x = M*seq(tr).x;
   end
 end
end