function seqnew = rebinRaster(seq,dt)
%
% function seq = rebinRaster(seq,dt)
%
% rebin seq by a factor of dt
%


Trials = numel(seq);
yDim   = size(seq(1).y,1);

if isfield(seq,'x')
   xDim = size(seq(1).x,1);
end

seqnew = seq;

for tr=1:Trials

  T    = size(seq(tr).y,2);
  Tnew = floor(T/dt);

  yold = reshape(seq(tr).y(:,1:Tnew*dt),yDim,dt,Tnew);
  %     ynew = squeeze(sum(yold,2)); % wrong result if yDim==1
  ynew = reshape(sum(yold,2), yDim, Tnew); %
  
  seqnew(tr).y = ynew;
  seqnew(tr).T = Tnew;

  if isfield(seq,'yr')
    yrold = reshape(seq(tr).yr(:,1:Tnew*dt),yDim,dt,Tnew);
    %        yrnew = squeeze(sum(yrold,2));
    yrnew = reshape(sum(yrold,2), yDim, Tnew);
    seqnew(tr).yr = yrnew;
  end

  if isfield(seq,'x')
    xold = reshape(seq(tr).x(:,1:Tnew*dt),xDim,dt,Tnew);
    %        xnew = squeeze(sum(xold,2));
    xnew = reshape(sum(xold,2), xDim, Tnew);
    seqnew(tr).x = xnew;
  end  
  
end 
