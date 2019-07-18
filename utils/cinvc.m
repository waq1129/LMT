% function [f,df,yf] = cinvc(cufx,invcc,invw,A,b,a)
%
% % cufx = reshape(cufx,2,[]);
% % invcc = inv(cufx*invw*cufx'+eye(2));
% f = A*cufx'*invcc*cufx*b;
% yf = a*A*cufx'*invcc*cufx*b;
% df = invcc*cufx*b*a*A+(a*A*cufx'*invcc)'*b'-invcc*(a*A*cufx')'*(cufx*b)'*invcc*cufx*invw-invcc*cufx*b*a*A*cufx'*invcc*cufx*invw;
% % df = vec(df);

function [f,yf1,df1,df2,df3] = cinvc(cufx,invcc,invw,b,a1,a2,a3)
if nargin<5
    a1 = 0;
    a2 = 0;
    a3 = 0;
end
% invw = diag(invw);
cb = cufx*b;
invwc = bsxfun(@times,invw,cufx');
invccb = invcc*cb;
invwinvc = invwc*invcc;
f = invwc*invccb;

if nargout >1
    %%
    yf1 = a1*f;
    ainvwinvc = a1*invwinvc;
    df1 = invccb*(a1.*invw')+ainvwinvc'*b'-(invwinvc'*a1')*(cb'*invwinvc')-invccb*(ainvwinvc*invwc');
    
    %%
    ainvwinvc = a2*invwinvc;
    df2 = invccb*(a2.*invw')+ainvwinvc'*b'-(invwinvc'*a2')*(cb'*invwinvc')-invccb*(ainvwinvc*invwc');
    
    %%
    ainvwinvc = a3*invwinvc;
    df3 = invccb*(a3.*invw')+ainvwinvc'*b'-(invwinvc'*a3')*(cb'*invwinvc')-invccb*(ainvwinvc*invwc');
end