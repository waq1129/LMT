function [dK_r2] = SE_cov_dK_h(x,z,ell,sf2,i1)
[r2, dr2_1] = r2_xz(x,z,ell,1,i1);
dK_r2 = sf2*exp(r2/2).*(dr2_1/2);
