function L = logmvnpdf(X,mu,C)
% L = logmvnpdf(X,mu,C)
%
% Evaluates log of mulivariate normal pdf with mean mu and covariance C 
% for each row of X

% Check if mu is a row vector and convert if so
if size(mu,1)==1
    mu = mu';
end

% Log-determinant term
logdettrm = -.5*logdet(2*pi*C);

% Quadratic term
Xctr = bsxfun(@minus,X,mu');  % centered X
Qtrm = -.5*sum(Xctr.*(Xctr/C),2);
L = Qtrm+logdettrm;

