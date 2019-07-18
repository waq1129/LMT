function [fun,pp,Mspline,splinePrs]=fit_spline_mse(x,y,ss)
% [fun,pp,Mspline,splinePrs] = fit_spline_mse(x,y,ss)
% 
%  Fit a function y = f(x) with a cubic spline, defined using a set of
%  breaks, smoothness and extrapolation criteria, by minimize MSE
%
% Inputs:
%   x,y - spline minimizes (f(x)-y).^2
%   ss - spline struct with fields:
%        .breaks - breaks between polynomial pieces
%        .smoothness - derivs of 1 less are continuous
%           (e.g., smoothness=3 -> 2nd derivs are continuous)
%        .extrapDeg - degree polynomial for extrapolation on ends
%
% Outputs:
%   fun - function handle for nonlinearity (uses 'ppval')
%   pp - piecewise polynomial structure
%   Mspline - matrix for converting params to spline coeffs
%   splinePrs -  vector x such that spline coeffs = Mspline*x
%
% last updated: 26/03/2012 (JW Pillow)


% Compute design matrix
[Xdesign,Ysrt,Mspline] = mksplineDesignMat(x,y,ss,1);

% Solve least-squares regression problem
splinePrs = Xdesign\Ysrt;

% Create function handle for resulting spline
[fun,pp] = mksplinefun(ss.breaks, Mspline*splinePrs); 
