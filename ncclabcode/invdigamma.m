function y=invdigamma(x)
% y = invdigamma(x)
%
% Inverse digamma (psi) function.  
%
% Digamma is the derivative of log(gamma(x)).  This calculates the value
% y > 0 for a value x such that digamma(y) = x.
%
% For x >> 0 : invdigamma(x) =~ exp(x)+.5
% For x << 0 : invdigamma(x) =~ -1/(x-.5)
%
% Uses initial guess based on large and small range values, then updates
% with Newton's method.
%
% Requires trigamma.m and digamma.m (Lightspeed toolbox).
%
% $Id$

% Compute Initial approximation (accurate to within 0.01)
a1 = .915; a2 = -.871; % some constants 
b1 = 2.1915; b2 = -3.387; c1 = -2.675;
w1 = exp(a1*x + b1);  w1 = w1./(1+w1);  % sigmoid #1
w1(isnan(w1))=1;
w2 = exp(a2*x + b2);  w2 = w2./(1+w2);  % sigmoid #2
w2(isnan(w2))=1;
y = exp(x)+.5*w1 - w2./min(c1,x+.5); % initial approximation

% Run NITER steps of Newton's method
% NITER= 1 : 10e-4 accuracy
% NITER= 2 : 10e-8 
% NITER= 3 : 10e-14
NITER = 3;  
ii = (x<15); % values above this are asymptotically close to exp(x)+.5
for j = 1:NITER
    dx = x(ii)-digamma(y(ii));
    dxdy = trigamma(y(ii));
    y(ii) = y(ii)+dx./dxdy;
end

    
