function fx = stableLogLogExp(x)

fx = zeros(size(x));

fx(x>5)          = log(x(x>5));
fx(x<(-5))       = x(x<(-5));
fx(abs(x)<=5)    = log(log(1+exp(x(abs(x)<=5))));
