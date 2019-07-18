% test_nudft.m
%
% Script for testing out non-uniform inverse DFT.

% Note: Turns out this is slightly incorrect compared to standard definition of
% NUFT, which involves a matrix multiplication for forward NUFFT, and a
% least squares inverse for reconstruction.  

n = 60;  % number of dimensions for true signal;
t = 0:n-1;

% make Fourier basis matrix
wvec = [0:ceil((n-1)/2), -floor((n-1)/2):-1]'; % Fourier frequencies
B = 1/sqrt(n)*exp(-1i*2*pi/n*(wvec*wvec'));  % DFT basis 

% Generate signal in Fourier domain
nsupp = 4;  % number of frequencies in support for true signal (including dc)
iisupp = [1:nsupp,n-nsupp+2:n]'; % indices for this support

fhwts = randn(nsupp,1)+[0; 1i*randn(nsupp-1,1)]; % weights for true signal in Fourier domain
fh = [fhwts; conj(flipud(fhwts(2:end)))]; % add complex conjugate for neg frequencies
f = real(B(:,iisupp)*fh);  % signal on integer lattice

% Now make signal with uneven sampling of same function in the interval [0 n];
npts = 70; % number of samples
ti = sort(rand(npts,1)*n); % sample times
Bgen = 1/sqrt(n)*exp(-1i*2*pi/n*(ti*wvec(iisupp)')); % basis 
fi = real(Bgen*fh);

plot(t,f,'o-',ti,fi,'r*')

%%  Now show how to create Fourier domain representation of this signal (easy!)

% Make basis for inverse Fourier transform from evenly spaced freqs to ti
Bsamp = 1/sqrt(n)*exp(-1i*2*pi/n*(ti*wvec')); % full basis for function 
fhhat = Bsamp\fi; % find least-squares representation of signal in this basis

% Show that the Fourier represenations match
subplot(211);
plot(wvec(iisupp), real(fh), wvec, real(fhhat), '*'); title('real component');
subplot(212);
plot(wvec(iisupp), imag(fh), wvec, imag(fhhat), '*'); title('imaginary component');


%% Now test out real-valued representation using code

Bnufft = realnufftbasis(ti,n,n);  % make basis
fihat = Bnufft'\fi; % find representation in Fourier basis
forig = realifft(fihat); % convert Fourier representation back to evenly-spaced time
clf;plot(t,f,ti,fi, '*',t,forig,'--');
legend('true on lattice','nu-sampled', 'recon on lattice');


%% try it using *only* frequencies present in true signal

[Bnufft2,wvec] = realnufftbasis(ti,n,nsupp*2-1); % make it with restricted set of freqs
fihat2 = zeros(n,1);
fihat2(iisupp) = Bnufft2'\fi;
forig2 = realifft(fihat2);
clf;plot(t,f, ti,fi, '*',t,forig,'--',t,forig2,'.');
legend('true on lattice','nu-sampled', 'recon1','recon2');
