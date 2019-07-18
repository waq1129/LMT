function s = zasoFx(zaso, fct, varargin)
% This may look stupid, but it is to reduce a very likely and hard to debug problem where one calls z1.fxsum(z2, fct) which is nonsensical. This function makes the interface more Python-like and requires only one reference to the object.
    s = zaso.fx(zaso, fct, varargin{:});
end
