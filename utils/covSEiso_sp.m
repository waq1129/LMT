function [B,kdiag,K] = covSEiso_sp(rho,len,x)
nd = size(x,2); % numer of dimenions
condthresh = 1e12;
minlen = len*0.8;
maxlen = len*1.2;
rho = rho^0.5;

% Set Fourier-domain prior for g, mu=0, K=kdiag
minl = minlen*ones(nd,1);% max([minlen, len]); % mininum length scale to consider
% set up Fourier frequencies
xwid = max(x)-min(x); %max([10, model.n]); %diff(model.bounds)*(model.gridsize/(model.gridsize-1)); % estimated support along each dimension
nxcirc = xwid(:)+4*(maxlen*ones(nd,1));%ceil(xwid'*1.25); % location of circular boundary
maxfreq = floor(nxcirc'./(pi*minl)*sqrt(.5*log(condthresh))); % max freq to use
% nmaxfreq = min([200, maxfreq]);
% resolution = maxfreq/nmaxfreq;
% nxcirc(i) = nxcirc(i)/resolution;
nw = maxfreq*2+1; % number of fourier frequencies
[Bmats, Bmatsdx, Bmatsddx, wwnrmvecs, kdiagvecs, kdiags] = ...
    realbasis_nD_curv_kron(x, len, rho, nw, nxcirc, nd, 0, condthresh, minl);
B = Bmats{1};
kdiag = kdiagvecs{1};
if nargout>2
    K = B'*diag(kdiag)*B;
end