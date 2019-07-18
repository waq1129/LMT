function xh = prunedfft(x,k)
% Pruned FFT - computes first k components of n-point dft of x
% 
% xh = prunefft(x,k)
%
% INPUT:
%  x [n x m] - input signal whose fft to compute
%  k [1 x 1] - number of dft components to return
%                
% OUTPUT:
%  xh [k x 1] - first k fourier components of x
% 
% Note: k must evenly divide n

% Compute sizes
[n,m] = size(x);
nstrd = n/k; % stride

% Compute twiddle factors
twids = (exp(-1i*2*pi/n*(bsxfun(@times,(0:nstrd-1)',(0:k-1)))));

% Compute FFT
if m==1
    xh  = sum(fft(reshape(x,nstrd,k),[],2).*twids,1).';
else
    xh  = squeeze(sum(bsxfun(...
        @times,fft(reshape(x,nstrd,k,[]),[],2),twids),1));
end
