function params = touchField(params,ftouch,fval);
%
% params = touchField(params,ftouch,fval);
%


if nargin<2.5
  fval = [];
end

if ~isfield(params,ftouch)
  params = setfield(params,ftouch,fval);
end
