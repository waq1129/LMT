function plotMatrixSpectrum(A,varargin)
%
% plot matrix spectra
% 
% (c) Lars Buesing, 2014

figh = -1;
col  = 'r';
linw  = 2.0;
linwc = 2.0;
assignopts(who,varargin);

if figh<0
  figure
else
  figure(figh)
end

hold on;
p=circle([0,0],1,1000,'k');
set(p,'LineWidth',linwc)

if (size(A,1)==1) || (size(A,2)==1)
  EigA = A;
else
  EigA = eig(A);
end

plot(real(EigA),imag(EigA),['x' col],'Linewidth',linw)
axis off;