function postm = get_tc(xx,ffmat,xgrid,rhoff,lenff)

ffTYPE = 2;
covfun = covariance_fun(rhoff,lenff,ffTYPE); % get the covariance function

[K0,dK] = covfun(xx,xx);
K = K0+1e-4*eye(size(xx,1));
K1 = covfun(xgrid,xx);

postm = K1*(K\ffmat);
