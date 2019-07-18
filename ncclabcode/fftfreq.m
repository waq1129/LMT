function w = fftfreq(N)
% Pass back the fft frequencies (useful for plotting spectrum). 
%
% INPUT:
%    N = # samples in signal
%
% OUTPUT:
%    w = frequencies of fft 
%
% Use 'fftshift' to rotate so that 0 is in center

w = [0:floor((N-1)/2), -ceil((N-1)/2):-1]';