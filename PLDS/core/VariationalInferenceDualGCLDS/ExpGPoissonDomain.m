function domFlag = ExpGPoissonDomain(lam, K)
% Yuanjun Gao 2015

if (min(lam)<0)||any(isnan(lam))||any(isinf(lam))
   domFlag = false;
else
   lamsum = sum(reshape(lam, K, []), 1);
   if(max(lamsum) > 1)
       domFlag = false;
   else
   domFlag = true;
   end
end
