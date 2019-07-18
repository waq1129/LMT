function [B,wvec] = realfftbasis(nx,nn,wvec)
% Basis of sines+cosines for nn-point discrete fourier transform (DFT)
%
% [B,wvec] = realfftbasis(nx,nn,w)
%
% For real-valued vector x, realfftbasis(nx,nn)*x is equivalent to realfft(x,nn) 
%
% INPUTS:
%  nx - number of coefficients in original signal
%  nn - number of coefficients for FFT (should be >= nx, so FFT is zero-padded)
%  wvec (optional) - frequencies: positive = cosine
%
% OUTPUTS:
%   B [nn x nx] or [nw x nx] - DFT basis 
%   wvec - frequencies associated with rows of B
%
% See also: realfft, realifft

if nargin < 2
    nn = nx;
end

if nargin < 3
    % Make frequency vector
    ncos = ceil((nn+1)/2); % number of cosine terms (positive freqs)
    nsin = floor((nn-1)/2); % number of sine terms (negative freqs)
    wvec = [0:(ncos-1), -nsin:-1]'; % vector of frequencies
end

% Divide into pos (for cosine) and neg (for sine) frequencies
wcos = wvec(wvec>=0); 
wsin = wvec(wvec<0);  

x = (0:nx-1)'; % spatial pixel indices
if ~isempty(wsin)
    B = [cos((wcos*2*pi/nn)*x'); sin((wsin*2*pi/nn)*x')]/sqrt(nn/2);
else
    B = cos((wcos*2*pi/nn)*x')/sqrt(nn/2);
end

% make DC term into a unit vector
izero = (wvec==0); % index for DC term
B(izero,:) = B(izero,:)./sqrt(2);  

% if nn is even, make Nyquist term (highest cosine term) a unit vector
if (nn/2 == max(wvec))
    ncos = ceil((nn+1)/2);
    B(ncos,:) = B(ncos,:)./sqrt(2);
end
