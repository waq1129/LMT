function s = miniBatchCompute(argMagic, zaso, fct, zasoIdx)
% uses zaso.nMiniBatch to obtain a sequential segment of samples.

% use the first sample to determine the size of the returned sample
switch argMagic
    case 1
	miniX = zaso.X(1);
	s = fct(miniX);
    case 2
	miniY = zaso.Y(1);
	s = fct(miniY);
    case 3
	miniX = zaso.X(1);
	miniY = zaso.Y(1);
	s = fct(miniX, miniY);
end % switch
assert(size(s, 2) == 1, 'function handle should return (d x #samples)');

if nargin < 4; zasoIdx = 1:zaso.N; end
if isempty(zasoIdx); return; end
s = zeros(size(s, 1), numel(zasoIdx));

for t = 1:zaso.nMiniBatch:numel(zasoIdx)
    zIdx = t:min(numel(zasoIdx), t + zaso.nMiniBatch - 1);
    idx = zasoIdx(zIdx);
    switch argMagic
	case 1
	    miniX = zaso.X(idx);
	    s(:, zIdx) = fct(miniX);
	case 2
	    miniY = zaso.Y(idx);
	    s(:, zIdx) = fct(miniY);
	case 3
	    miniX = zaso.X(idx);
	    miniY = zaso.Y(idx);
	    s(:, zIdx) = fct(miniX, miniY);
    end % swtich
end % for

end % miniBatchCompute
