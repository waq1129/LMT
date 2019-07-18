function [f df] = ExpFamPCAlinkFuncCost(CXd,Y,xDim,lambda,s,CposConstrain,linkFunc,dlinkFunc)
%
% [f df] = ExpFamPCACost(CXd,Y,xDim,lambda)
%
% (c) L Buesing 2014

[yDim T] = size(Y);


d  = CXd(end-yDim+1:end);
CX = reshape(CXd(1:end-yDim),yDim+T,xDim);
C  = CX(1:yDim,:);
if CposConstrain; C = exp(C); end;
X  = CX(yDim+1:end,:)';

nu = bsxfun(@plus,C*X+s,d);
Yhat = linkFunc(nu);

f = sum(vec(-Y.*log(Yhat)+Yhat))+lambda/2*(norm(C,'fro')^2+norm(X,'fro'));

YhatmY = (Yhat-Y).*(dlinkFunc(nu)./Yhat);

gX = C'*YhatmY+lambda*X;
gC = YhatmY*X'+lambda*C;
if CposConstrain; gC = gC.*C; end
gd = sum(YhatmY,2);

df = [vec([gC;gX']);gd];

