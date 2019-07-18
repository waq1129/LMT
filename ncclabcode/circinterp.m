function G = circinterp(F, mu)
% G = circinterp(F, mu)
%
% Shifts a matrix F circularly by mu units (mod 2pi) using
% its fft (shifting the phase of each component), where a shift
% pi corresponds to moving halfway around the x axis. 
%
% Positive mu gives downward / right shifts.
% 
% For matrix inputs, operates along columns


[n,m] = size(F);
rowvec = 0;
if (n == 1)
    rowvec = 1;
    F = F';
    [n,m] = size(F);
end

if mod(n,2) == 1
    w = (-(n-1)/2:1:(n-1)/2)';
else
    w = (-n/2:1:n/2-1)';
end

Fhat = fftshift(fft(F));
thets = exp(-1i*mu*w);
Frot = Fhat.*repmat(thets,1,m);
G = real(ifft(ifftshift(Frot)));

if rowvec
    G = G';
end
