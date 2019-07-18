function plotPosterior(seq,trId,params)
%
%
%

[xDim T] = size(seq(trId).posterior.xsm);

try
	Pidx = [1 1+params.dE];
catch
	Pidx = [1 2];
end

if nargin<1.5 
  trId = 1;
end

xsm  = seq(trId).posterior.xsm;
xerr = zeros(size(xsm));
for t=1:T
  xidx = (t-1)*xDim+1:t*xDim;
  xerr(:,t) = sqrt(diag(seq(trId).posterior.Vsm(xidx,:)));
end

figure; hold on; title('posterior')

for i=1:numel(Pidx)
  subplot(numel(Pidx),1,i); hold on;
  pidx = Pidx(i);
  errorbar(1:T,xsm(pidx,:),xerr(pidx,:),'r')
  try;plot(1:T,seq(trId).x(pidx,:),'linewidth',2);end;
  plot(1:T,xsm(pidx,:),'r','linewidth',2)
  ylabel('x(t)');  
  if i==numel(Pidx);xlabel('t');end
  %figSize = {14,10};
  %figuresize(figSize{:},'centimeters')
end

if isfield(seq(1).posterior,'phi')
try
  figure
  plot(seq(1).posterior.phi,'x','MarkerSize',10,'linewidth',2)
  xlabel('neuron no');ylabel('phi')
  %figSize = {14,10};
  %figuresize(figSize{:},'centimeters')

%{
  figure
  piNow = exp(seq(1).posterior.phi);
  piNow = bsxfun(@times,piNow,1./sum(piNow,2));
  plot(piNow,'x','MarkerSize',10,'linewidth',2)
  %plot(seq(1).posterior.phi,'rx','MarkerSize',10,'linewidth',2)
  xlabel('neuron no');ylabel('pi')
  %figSize = {14,10};
  %figuresize(figSize{:},'centimeters')
%}
end
end
