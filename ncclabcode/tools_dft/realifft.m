function x = realifft(xhat,nx)
% Inverse fourier transform from representation in terms of sines and cosines
% 
% x = realifft(xhat,nx)
%
% Compute version of inverse Discrete Fourier Transform (DFT) using ifft,  
% but with cosine and sine terms separately so that input vector is real-valued.  
% 
% Let n denote the length of xhat.
% For even n, input signal organized as:
% [DC(0), cos(1), ... cos(n/2), sin(-(n+2)/2), ... sin(-1)]';
%
% For odd n, input signal organized as:
% [DC(0), cos(1), ... cos((n-1)/2), sin(-(n-1)/2), ... sin(-1)]';
%
% Optional second argument assumes original signal was only length nx, and
% this representation was created by an n-point fft (with n > nx).
% 
% INPUT:
% ------
%  xhat - vector or matrix of Fourier transform
%    nx - # points in original real-valued signal (optional)
%
% OUTPUT:
% -------
%     x - inverse DFT of xhat
%
% See also: realifft


% Check if xhat is a row vector, convert to column if so
if (size(xhat,1) == 1), ROWVEC = true; xhat = xhat'; 
else ROWVEC = false;
end

% convert x to column if a row vector
if (size(xhat,1) == 1) 
    xhat = xhat';
end

% Make frequency vector
nxh = size(xhat,1); % number of coefficients in DFT
ncos = ceil((nxh+1)/2); % number of cosine terms
nsin = floor((nxh-1)/2); % number of sin terms

% Fix amplitude of DC term & Nyquist term (highest cosine term) if nxh even
xhat(1,:) = xhat(1,:)*sqrt(2);
if (mod(nxh,2) == 0), xhat(ncos,:) = xhat(ncos,:)*sqrt(2); % Nyquist term
end

% Insert real terms
xfft = xhat;
xfft(ncos+1:end,:) = flipud(xhat(2:nsin+1,:));
% Insert imaginary terms
xfft(2:nsin+1,:) = xfft(2:nsin+1,:) + 1i*flipud(xhat(ncos+1:end,:));
xfft(ncos+1:end,:) = xfft(ncos+1:end,:) - 1i*xhat(ncos+1:end,:);

% Take inverse fourier transform
x = real(ifft(xfft))*sqrt(nxh/2);

% Check if nx passed in and truncate if necessary
if nargin > 1
    x = x(1:nx,:);
end

% Convert back to row vec, if necessary
if ROWVEC,  x = x';  
end
