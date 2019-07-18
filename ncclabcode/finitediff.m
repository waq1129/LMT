function dx = finitediff(x,dt)
% FINITEDIFF - compute centered estimate of derivative dx/dt 
%
% dx = finitediff(x,dt);
%
% Averaging forward and backward derivatives via diff(x)/dt
%
% Inputs:
%      x Nx1 - vector of measurements 
%     dt 1x1 - time step
%
% Outputs:
%     dx Nx1 - vector of dx/dt

if (nargin == 1)
    dt = 1;
end

[m,n] = size(x);
if m==1  % Check if row vector passed in
    x = x';
    isrow = 1;
else
    isrow = 0;
end

dx1 = diff(x);
dx = .5*([dx1(1,:);dx1]+[dx1;dx1(end,:)])/dt;

if isrow  % Convert back to row vector, if row passed in
    dx = dx';
end
