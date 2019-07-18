function [xhat,wvec] = realfft(x, n)
% Compute n-point fft and store sine and cosine terms separately to avoid complex values
%
% [xhat,wvec] = realfft(x, nxcirc);
%
% Computes n-point orthogonal discrete Fourier transform (so norm of x does not
% change) and represents cos and sin terms separately to avoid the use of
% complex values.
% 
% For even n, output organized as:
% [DC(0), cos(1), ... cos(n/2), sin(-(n+2)/2), ... sin(-1)]';
%
% For odd n, output organized as:
% [DC(0), cos(1), ... cos((n-1)/2), sin(-(n-1)/2), ... sin(-1)]';
%
% INPUT:
% ------
%     x - vector, matrix, or tensor (operates along columns, if matrix or tensor)
%     n - number of points in dft (so x padded with zeros if necessary)
%
% OUTPUT:
% ------
%  xhat - dft of x 
%  wvec - vector of Fourier frequencies, ordered as above for even/odd n
%
% See also: realifft


% Check if x is a row vector, convert to column if so
if (size(x,1) == 1), ROWVEC = true; x = x'; 
else ROWVEC = false;
end

% Check if n passed in
if nargin == 1,  n = size(x,1);
end

% Compute fast fourier transform
xfft = fft(x,n)/sqrt(n/2); 

% Fix norm of DC term 
xfft(1,:) = xfft(1,:)/sqrt(2);
% If even, fix nyquist term
if mod(n,2) == 0 
    imx = ceil((n+1)/2);
    xfft(imx,:) = xfft(imx,:)/sqrt(2);
end

% Extract real (cos) and imaginary (sin) terms
xhat = real(xfft);
isin = ceil((n+3)/2); % index where sin terms start
xhat(isin:end,:) = -imag(xfft(isin:end,:));

% Convert back to row vec, if necessary
if ROWVEC,  xhat = xhat';  
end

% Compute frequency vector, if desired
if nargout > 1
    ncos = ceil((n+1)/2); % number of negative frequencies;
    nsin = floor((n-1)/2); % number of positive frequencies;
    wvec = [0:(ncos-1), -nsin:-1]'; % vector of frequencies
end