function [xhat,colfreqs,rowfreqs] = realfft2(x,ncol,nrow)
% Fast 2D discrete fourier transform, represented with real-valued coeffs
% 
% [xhat,colfreqs,rowfreqs] = realfft2(x,ncol,nrow)
%
% Computes n1-point fft along columns, and n2-point fft along rows, and
% uses separate terms for cos and sin terms to avoid complex values.
% See 'realfft.m' for information about frequency vectors
%
% Note: implemented with two calls to realfft, which has the consequence
% that we mix together (wx+wy) and (wx-wy) terms, losing information about
% off-diagonal orientation in the Fourier domain. So, use only with
% separable kernels.
%
% See also: realfft, realifft, realfftbasis

% Check that input is not a vector
if (size(x,1) == 1) || (size(x,2)==1)
    
    warning('Input is a vector: calling realfft');
    [xhat,colfreqs] = realfft(x);
    rowfreqs = 0;
    if size(xhat,1) == 1
        % check if row vector, swap col and row freqs if so
        rowfreqs = colfreqs;
        colfreqs = 0;
    end

elseif nargin == 1
    % Use non-padded FFT
    
    % First compute real fft along columns
    [xhat0, colfreqs] = realfft(x);
    
    % Now do it along rows
    [xhat, rowfreqs] = realfft(permute(xhat0,[2,1,3]));
    xhat = permute(xhat,[2,1,3]);  % Change cols back to rows

elseif nargin == 3
    % Use padded FFT

    % First compute real fft along columns
    [xhat0, colfreqs] = realfft(x,ncol);
    
    % Now do it along rows
    [xhat, rowfreqs] = realfft(permute(xhat0,[2,1,3]),nrow);
    xhat = permute(xhat,[2,1,3]);  % Change cols back to rows
    
else
    
    error('Should have 1 or 3 arguments: realfft2(x,ncol,nrow)');

end
