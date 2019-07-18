function domFlag = ExpPoissonDomain(lam);
%

if (min(lam)<0)||any(isnan(lam))||any(isinf(lam))
   domFlag = false;
else
   domFlag = true;
end
