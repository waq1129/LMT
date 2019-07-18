function zaso = encapsulateRaw(X, Y, bX, bY, isBulkProcess)
% zaso = encapsulateRaw(X, Y, bX, bY, isBulkProcess)
% reference implementation ZASO for no feature space transformation
% Use this file as a template to create a new custom encapsuation procedure.
%
% Input:
%   X: (N x dx) N samples from the indepenent variables
%   Y: (N x dy) corresponding samples from the dependent variables
%   bX: @(X(idx,:)) -> (numel(idx) x dimx) feature transformation
%	bX can take cell array, in which case X should be a cell array
%	bX must return consistent feature dimension
%   bY: @(Y(idx,:)) -> (numel(idx) x dimy) feature transformation similar to bX
%   isBulkProcess: (logical/optional) defaut: false
%	If false, basic behavior is mini-batch computation.
%	If true, full matrix is used all the time, and zasoIdx arguments are
%	all ignored.
%
% Output:
%   zaso: ZASO!
%
% $Id$

[nx, mx] = size(X);
[ny, my] = size(Y);
assert(nx == ny, '# of samples mismatch');
assert(nx > 0, 'must have at least 1 sample');
zaso.N = nx;
zaso.Nsub = zaso.N;
zaso.sub2idx = @(x) x;

if nargin > 2 && ~isempty(bX)
    if isa(bX, 'function_handle')
	zaso.bX = bX;
	zaso.X = @(idx) bX(X(idx,:));
	zaso.dimx = size(bX(X(1,:)), 2);
    else
	error('bX must be a function_handle');
    end
else
    zaso.X = @(idx) X(idx,:);
    zaso.dimx = mx;
end

if nargin > 3 && ~isempty(bY)
    if isa(bY, 'function_handle')
	zaso.bY = bY;
	zaso.Y = @(idx) bY(Y(idx,:));
	zaso.dimy = size(bY(Y(1,:)), 2);
    else
	error('bY must be a function_handle');
    end
else
    zaso.Y = @(idx) Y(idx,:);
    zaso.dimy = my;
end

if nargin < 5 || isempty(isBulkProcess)
    zaso.isBulkProcess = false;
else
    zaso.isBulkProcess = isBulkProcess;
end

zaso.desc = sprintf('%s(%s)', mfilename, datestr(now, 30));
zaso.Id = '$Id$';

if zaso.isBulkProcess
    % save the full matrix X and Y for all indices
    if isfield(zaso, 'bX')
	zaso.Xbulk = zaso.bX(X);
    else
	zaso.Xbulk = X;
    end

    if isfield(zaso, 'bY')
	zaso.Ybulk = zaso.bY(Y);
    else
	zaso.Ybulk = Y;
    end

    zaso.fxsum = @(varargin) bulkSum(1, varargin{:});
    zaso.fysum = @(varargin) bulkSum(2, varargin{:});
    zaso.fxysum = @(varargin) bulkSum(3, varargin{:});

    zaso.fx = @(varargin) bulkCompute(1, varargin{:});
    zaso.fy = @(varargin) bulkCompute(2, varargin{:});
    zaso.fxy = @(varargin) bulkCompute(3, varargin{:});

    zaso.farray  = @bulkArray;
else
    % use miniBatch to compute things
    zaso.nMiniBatch = 1024 * 8;

    zaso.fxsum = @(varargin) miniBatchSum(1, varargin{:});
    zaso.fysum = @(varargin) miniBatchSum(2, varargin{:});
    zaso.fxysum = @(varargin) miniBatchSum(3, varargin{:});

    zaso.fx = @(varargin) miniBatchCompute(1, varargin{:});
    zaso.fy = @(varargin) miniBatchCompute(2, varargin{:});
    zaso.fxy = @(varargin) miniBatchCompute(3, varargin{:});

    zaso.farray  = @miniBatchArray;
end
end % encapsulateRaw

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s = bulkCompute(argMagic, zaso, fct, zasoIdx)
% CAUTION: zasoIdx is completely ignored!

if nargin > 3; warning('zasoIdx is ignored in Bulk mode!\n'); end

switch argMagic
    case 1
	s = fct(zaso.Xbulk);
    case 2
	s = fct(zaso.Ybulk);
    case 3
	s = fct(zaso.Xbulk, zaso.Ybulk);
end % switch
end % bulkSum

function s = bulkSum(argMagic, zaso, fct, zasoIdx)
% CAUTION: zasoIdx is completely ignored!

if nargin > 3; warning('zasoIdx is ignored in Bulk mode!\n'); end

switch argMagic
    case 1
	s = fct(zaso.Xbulk);
    case 2
	s = fct(zaso.Ybulk);
    case 3
	s = fct(zaso.Xbulk, zaso.Ybulk);
end % switch
end % bulkSum

function [sum_result, agg_result] = bulkArray(zaso, fct_sum, fct_agg, zasoIdx)
% CAUTION: zasoIdx is completely ignored!

if nargin > 3; warning('zasoIdx is ignored in Bulk mode!\n'); end

agg_result = cell(numel(fct_agg), 1);
sum_result = cell(numel(fct_sum), 1);

for k = 1:numel(fct_agg)
    agg_result{k} = feval(fct_agg{k}, zaso.Xbulk, zaso.Ybulk);
end

for k = 1:numel(fct_sum)
    sum_result{k} = feval(fct_sum{k}, zaso.Xbulk, zaso.Ybulk);
end
end
