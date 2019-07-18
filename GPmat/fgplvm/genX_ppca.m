function X = genX_ppca(q, d, Y, options)

% FGPLVMCREATE Create a GPLVM model with inducing variables.
% FORMAT
% DESC creates a GP-LVM model with the possibility of using
% inducing variables to speed up computation.
% ARG q : dimensionality of latent space.
% ARG d : dimensionality of data space.
% ARG Y : the data to be modelled in design matrix format (as many
% rows as there are data points).
% ARG options : options structure as returned from
% FGPLVMOPTIONS. This structure determines the type of
% approximations to be used (if any).
% RETURN model : the GP-LVM model.
%
% COPYRIGHT : Neil D. Lawrence, 2005, 2006
%
% MODIFICATIONS : Carl Henrik Ek, 2010
%
% SEEALSO : modelCreate, fgplvmOptions

% FGPLVM

if size(Y, 2) ~= d
    error(['Input matrix Y does not have dimension ' num2str(d)]);
end

if isstr(options.initX)
    initFunc = str2func([options.initX 'Embed']);
    X = initFunc(Y, q);
else
    if size(options.initX, 1) == size(Y, 1) ...
            & size(options.initX, 2) == q
        X = options.initX;
    else
        error('options.initX not in recognisable form.');
    end
end
