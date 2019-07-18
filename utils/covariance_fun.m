function covfun = covariance_fun(rhoff,lenff,ffTYPE)
kSE = @(rho,len,x,z) covSEiso_deriv([log(len); log(rho)/2], x, z);
kAR1 = @(rho,len,x,z) covAR1iso([log(len); log(rho)/2], x, z);

switch ffTYPE
    case 1
        % covfun = @(x1,x2)(rhoff*exp(-abs(bsxfun(@minus,x1,x2'))/lenff)); % cov fun
        covfun = @(x1,x2) kAR1(rhoff,lenff,x1,x2); % cov fun
    case 2
        % covfun = @(x1,x2)(rhoff*exp(-bsxfun(@minus,x1,x2').^2/(2*lenff^2))); % cov fun
        covfun = @(x1,x2) kSE(rhoff,lenff,x1,x2); % cov fun
        
end
