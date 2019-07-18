function zaso = encapsulateAbstract(xHandle, yHandle, N)
% Most generic mini-batch implementation of ZASO. Two function handles that
% return data given integer indices is encapsulated.
% This is useful when the responses are generated on the fly.
%
% Input
%   xHandle: @(idx) -> [numel(idx) x dimx]
%   yHandle: @(idx) -> [numel(idx) x dimy]
%   N: maximum index allowed for each handle
%
% Output
%   zaso: ZASO!
%
% $Id$

assert(N > 0 && mod(N, 1) == 0, 'N must be a positive integer');
zaso.N = N;
zaso.Nsub = N;
zaso.sub2idx = @(x) x;
assert(isa(xHandle, 'function_handle'), 'xHandle must be a function handle');
assert(isa(yHandle, 'function_handle'), 'yHandle must be a function handle');
zaso.X = xHandle;
zaso.Y = yHandle;
zaso.dimx = size(zaso.X(1), 2);
assert(size(zaso.X(1), 1) == 1, 'xHandle must return a vector of consistent size');
assert(zaso.dimx == size(zaso.X(N), 2), 'dimension mismatch');
zaso.dimy = size(zaso.Y(1), 2);
assert(size(zaso.Y(1), 1) == 1, 'yHandle must return a vector of consistent size');
assert(zaso.dimy == size(zaso.Y(N), 2), 'dimension mismatch');
zaso.desc = sprintf('%s(%s)', mfilename, datestr(now, 30));

zaso.nMiniBatch = 1024 * 8;

zaso.fxsum = @(varargin) miniBatchSum(1, varargin{:});
zaso.fysum = @(varargin) miniBatchSum(2, varargin{:});
zaso.fxysum = @(varargin) miniBatchSum(3, varargin{:});

zaso.fx = @(varargin) miniBatchCompute(1, varargin{:});
zaso.fy = @(varargin) miniBatchCompute(2, varargin{:});
zaso.fxy = @(varargin) miniBatchCompute(3, varargin{:});

zaso.farray  = @miniBatchArray;
