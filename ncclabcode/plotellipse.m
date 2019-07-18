function [h,el] = plotellipse(mu, covmat, r, varargin);
% h = plotellipse(mu, covmat, stdev, varargin);
% plots an ellipse of constant probability (i.e. a contour of
% constant deviation stdev) for a given bivariate gaussian distr.
% Inputs:
%    mu = column vector with the mean of the distr.
%    covmat = 2x2 covariance matrix
%    stdev = the standard deviation contour we wish to plot (e.g. 1, 2, .2, etc)
% Output: 
%    h = handle to plotted line
%    el = 100x2 matrix where the two columns provide the x and y values of the 
%         ellipse
%
% $Id$

[U, S, V] = svd(covmat);
thet = [0:(2*pi)/99:(2*pi+.0001)];
el = [r*U*sqrt(S)*[cos(thet); sin(thet)]]';
el(:,1) = el(:,1)+mu(1);
el(:,2) = el(:,2)+mu(2);
h = plot(el(:,1), el(:,2), varargin{:});
