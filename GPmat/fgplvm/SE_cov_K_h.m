function [K] = SE_cov_K_h(x,z,ell,sf2)
r2 = r2_xz(x,z,ell);

K = sf2*exp(r2/2);
