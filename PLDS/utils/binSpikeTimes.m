function seq = binSpikeTimes(spikeTimes,dt,varargin);
%
% seq = binSpikeTimes(spikeTimes,dt,varargin)
%
% convert data in noise-workshop data format to seq struct
%

Tmax = [];

assignopts(who,varargin);

yDim = numel(spikeTimes);
seq = [];


Trials = numel(spikeTimes{1});

for tr=1:Trials
  
  if isempty(Tmax)
    Tall = [];
    for yd=1:yDim
      Tall = [Tall vec(spikeTimes{yd}{tr})'];
    end
    T = ceil(max(Tall));
  else
    T = Tmax;
  end
  
  bins = 0:dt:T;
  
  seq(tr).y = zeros(yDim,numel(bins));
  
  for yd=1:yDim
    if ~isempty(spikeTimes{yd}{tr})
      seq(tr).y(yd,:) = histc(spikeTimes{yd}{tr},bins)';
    end
  end
  seq(tr).y = seq(tr).y(:,1:end-1);
  seq(tr).T = size(seq(tr).y,2);
  
  
end
