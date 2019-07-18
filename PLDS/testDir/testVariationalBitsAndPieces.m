clear all
close all


xDim   = 10;
yDim   = 15;
T      = 10;
Trials = 50;


trueparams = PLDSgenerateExample('T',T,'Trials',Trials,'xDim',xDim,'yDim',yDim,'doff',-2.0);
seq = PLDSsample(trueparams,T,Trials);

VarInfparams.y = seq(1).y;
VarInfparams.A = trueparams.model.A;
VarInfparams.C = trueparams.model.C;
Cl = {}; for t=1:T; Cl = {Cl{:} VarInfparams.C}; end
VarInfparams.W = sparse(blkdiag(Cl{:}));
VarInfparams.WlamW  = sparse(zeros(xDim*T));


[yDim xDim] = size(VarInfparams.C);

VarInfparams.CC = zeros(xDim,xDim,yDim);
for yy=1:yDim
  VarInfparams.CC(:,:,yy) = VarInfparams.C(yy,:)'*VarInfparams.C(yy,:);
end
VarInfparams.CC = reshape(VarInfparams.CC,xDim^2,yDim);

lamTest = rand(yDim*T,1)+0.1;


WlamWFull = VarInfparams.W'*diag(lamTest)*VarInfparams.W;

for t=1:T
  xidx = ((t-1)*xDim+1):(t*xDim);
  yidx = ((t-1)*yDim+1):(t*yDim);
  %VarInfparams.WlamW(xidx,xidx) = VarInfparams.C'*diag(lam(yidx))*VarInfparams.C; %debug-line, use below
  VarInfparams.WlamW(xidx,xidx) = reshape(VarInfparams.CC*lamTest(yidx),xDim,xDim);
end

max(abs(vec(WlamWFull-VarInfparams.WlamW)))./max(abs(vec(WlamWFull)))
%figure([WlamWFull VarInfparams.WlamW])

Lambda= buildPriorPrecisionMatrixFromLDS(trueparams,T);
Alam  = Lambda+VarInfparams.WlamW;
abs(logdet_plds(Alam,'chol')-log(det(Alam)))./abs(log(det(Alam)))


Vtest   = randn(xDim); Vtest = (Vtest+Vtest')/2;
CVCFull = diag(VarInfparams.C*Vtest*VarInfparams.C');
CVC     = VarInfparams.CC'*vec(Vtest);

max(abs(CVC-CVCFull))./max(abs(CVC))
