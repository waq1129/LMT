function s = miniBatchSum(argMagic, zaso, fct, zasoIdx)
% uses zaso.nMiniBatch to obtain a sequential segment of samples.

% use the first sample to 
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
s = zeros(size(s));

if nargin < 4; zasoIdx = 1:zaso.N; end
if isempty(zasoIdx); s = 0; return; end

for t = 1:zaso.nMiniBatch:numel(zasoIdx)
    zIdx = t:min(numel(zasoIdx), t + zaso.nMiniBatch - 1);
    idx = zasoIdx(zIdx);
    switch argMagic
	case 1
	    miniX = zaso.X(idx);
	    s = s + fct(miniX);
	case 2
	    miniY = zaso.Y(idx);
	    s = s + fct(miniY);
	case 3
	    miniX = zaso.X(idx);
	    miniY = zaso.Y(idx);
	    s = s + fct(miniX, miniY);
    end % switch
end % for

end % miniBatchSum
