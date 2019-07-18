function [f df] = ExpFamPCACost(CXd,Y,xDim,lambda,s,CposConstrain)
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
Yhat = exp(nu);

f = sum(vec(-Y.*nu+Yhat))+lambda/2*(norm(C,'fro')^2+norm(X,'fro'));

YhatmY = Yhat-Y;

gX = C'*YhatmY+lambda*X;
gC = YhatmY*X'+lambda*C;
if CposConstrain; gC = gC.*C; end
gd = sum(YhatmY,2);

df = [vec([gC;gX']);gd];

