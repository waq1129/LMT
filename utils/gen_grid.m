function tgrid = gen_grid(gridends,ng,nc)
varargin = cell(nc,1);
varargout = cell(nc,1);
for ii=1:nc
    varargin{ii} = linspace(gridends(ii,1),gridends(ii,2),ng);
end
[varargout{1:nc}] = ndgrid(varargin{:});
tgrid = []; for ii=1:nc tgrid = [tgrid vec(varargout{ii})]; end