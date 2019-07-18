function fh = errorbarFill(fillX, fillmean, fillwidth, varargin)
% Make "filled" error bar region (alternative to  'errorbar');
%
% fh = errorbarFill(fillX, fillmean, fillwidth, fillcolor, {plot args})
%
% You can use this as a replacement for errorbar(fillX, fillmean, fillwidth)
% except that errorbarFill does not plot the mean for you, since you may
% want to overlay all the area plots before plotting the means
%
% Input
%   fillX: x positions
%   fillmean: center of the error bar area
%   fillwidth: half-width of the error bar area
%   fillcolor: color of the fill 
%	           'r','g','b','c','m','y','w','k', or [r g b]
%
% Output
%   fh: handle for the polygon (fill object)
%
% Example call:
%  errorbarFill(x,y,yEB,.7*[1 1 1],'EdgeColor','none','FaceAlpha', 0.5);

assert(min(size(fillX)) == 1);
assert(min(size(fillmean)) == 1);
assert(min(size(fillwidth)) == 1);

if isempty(varargin)
    varargin = {.5*[1 1 1], 'EdgeColor', 'none', 'FaceAlpha', 0.5};
end

fillX = fillX(:)'; fillmean = fillmean(:)'; fillwidth = fillwidth(:)';
fh = fill([fillX fliplr(fillX)], [fillmean(:)' fliplr(fillmean(:)')] + [fillwidth -fliplr(fillwidth)], varargin{:});
